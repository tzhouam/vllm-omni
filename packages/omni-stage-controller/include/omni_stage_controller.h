// SPDX-License-Identifier: Apache-2.0
#pragma once

// Control-only C++17 subset of omni_stage_contracts.ReferenceStageController.
// The backend owns tensors, weights and state. A successful cancel/release ACK
// must come from that backend after hardware work and transfers have stopped.

#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace omni {

struct BackendIdentity {
    std::string backend, instance_id, worker_generation;
};

struct StateHandle {
    std::string session_id, backend, artifact_id;
    std::uint64_t layout_version = 0, epoch = 0;
    bool replayable = false, migratable = false;
    std::string backend_instance_id, worker_generation, state_id;

    auto identity() const {
        return std::tie(session_id, backend, artifact_id, layout_version, epoch,
                        replayable, migratable, backend_instance_id, worker_generation, state_id);
    }
    bool operator==(const StateHandle& other) const { return identity() == other.identity(); }
};

struct StageRequest {
    std::string request_id;
    std::uint64_t stage_id = 0, epoch = 0;
    std::string worker_generation, artifact_id, session_id;
    std::uint64_t state_layout_version = 0;
    std::optional<StateHandle> state;
};

struct StageEvent {
    std::string request_id;
    std::uint64_t stage_id = 0, epoch = 0, seq = 0;
    std::string kind, worker_generation;
    bool terminal = false;
    std::optional<StateHandle> state;
    std::uint64_t started_monotonic_ns = 0, emitted_monotonic_ns = 0;
    std::uint64_t input_watermark = 0, payload_nbytes = 0;
    std::string release_token;
};

enum class EmitResult { accepted, backpressure, retired };

class StageController {
public:
    StageController(std::map<std::uint64_t, BackendIdentity> backends,
                    std::uint64_t max_events, std::uint64_t max_bytes)
        : backends_(std::move(backends)), max_events_(max_events), max_bytes_(max_bytes) {
        if (backends_.empty() || !max_events_ || !max_bytes_) invalid("invalid controller bounds");
        for (const auto& entry : backends_) {
            const auto& backend = entry.second;
            if (backend.backend.empty() || backend.instance_id.empty() || backend.worker_generation.empty())
                invalid("backend identity is required");
        }
    }

    void begin(const StageRequest& request) {
        std::lock_guard<std::mutex> guard(lock_);
        if (active_ && (!producer_done_ || !pending_.empty())) invalid("request or payload is still active");
        if (!releasing_.empty()) invalid("backend state release is unresolved");
        if (request.request_id.empty() || request.artifact_id.empty() || request.session_id.empty()
                || !request.state_layout_version) invalid("v2 request identity is required");
        const auto backend = backends_.find(request.stage_id);
        if (backend == backends_.end() || request.worker_generation != backend->second.worker_generation)
            invalid("request worker generation or stage mismatch");
        const RequestKey key(request.request_id, request.stage_id, request.epoch, request.worker_generation);
        if (seen_requests_.count(key)) invalid("request identity was already used");
        if (request.state) {
            validate_state(request, *request.state);
            const auto state_key = key_for(request.stage_id, *request.state);
            const auto state = states_.find(state_key);
            if (state == states_.end() || !(state->second == *request.state) || retired_.count(state_key))
                invalid("continuation state is not live");
        }
        active_ = request;
        seen_requests_.insert(key);
        queue_.clear(); pending_.clear(); delivered_.clear(); seen_tokens_.clear();
        next_seq_ = 1; last_watermark_ = 0; used_bytes_ = 0;
        terminal_ = false; cancelled_ = false; producer_done_ = false;
    }

    EmitResult try_emit(const StageEvent& event) {
        std::lock_guard<std::mutex> guard(lock_);
        if (!active_ || cancelled_ || terminal_ || producer_done_) return EmitResult::retired;
        const auto& request = *active_;
        if (event.request_id != request.request_id || event.stage_id != request.stage_id
                || event.epoch != request.epoch || event.worker_generation != request.worker_generation)
            invalid("event request, stage, worker or epoch mismatch");
        if (event.seq != next_seq_ || event.input_watermark < last_watermark_)
            invalid("event sequence or watermark mismatch");
        if (event.kind.empty() || event.release_token.empty() || seen_tokens_.count(event.release_token)
                || !event.started_monotonic_ns || event.emitted_monotonic_ns < event.started_monotonic_ns)
            invalid("event needs timestamps and a unique acknowledgement token");
        if (event.payload_nbytes > max_bytes_) invalid("event exceeds admitted byte bound");
        if (event.state) validate_state(request, *event.state);
        if (pending_.size() >= max_events_ || event.payload_nbytes > max_bytes_ - used_bytes_)
            return EmitResult::backpressure;
        if (event.state) {
            const auto key = key_for(request.stage_id, *event.state);
            states_.insert_or_assign(key, *event.state);
            active_->state = event.state;
        }
        pending_.emplace(event.release_token, event.payload_nbytes);
        seen_tokens_.insert(event.release_token);
        used_bytes_ += event.payload_nbytes;
        ++next_seq_; last_watermark_ = event.input_watermark; terminal_ = event.terminal;
        queue_.push_back(event);
        return EmitResult::accepted;
    }

    std::optional<StageEvent> receive(const StageRequest& request) {
        std::lock_guard<std::mutex> guard(lock_);
        require_request(request);
        if (queue_.empty()) return std::nullopt;
        auto event = queue_.front(); queue_.pop_front();
        delivered_.insert(event.release_token);
        return event;
    }

    bool acknowledge(const StageRequest& request, const std::string& token) {
        std::lock_guard<std::mutex> guard(lock_);
        if (!same_request(request) || !delivered_.count(token)) return false;
        release_credit(token);
        return true;
    }

    void finish(const StageRequest& request) {
        std::lock_guard<std::mutex> guard(lock_);
        require_request(request);
        if (!cancelled_ && !terminal_) {
            retire_active_state();
            producer_done_ = true;
            invalid("backend stopped without a terminal event");
        }
        producer_done_ = true;
    }

    void cancel(const StageRequest& request) {
        std::lock_guard<std::mutex> guard(lock_);
        require_request(request);
        cancelled_ = true;
        retire_active_state();
        while (!queue_.empty()) {
            release_credit(queue_.front().release_token);
            queue_.pop_front();
        }
        // Delivered bytes remain charged. finish() is a separate backend
        // quiescence ACK; a timeout must not invoke it on the backend's behalf.
    }

    std::string request_release(std::uint64_t stage_id, const StateHandle& state) {
        std::lock_guard<std::mutex> guard(lock_);
        if (active_ && (!producer_done_ || !pending_.empty())) invalid("state still has work or payloads");
        const auto key = key_for(stage_id, state);
        const auto found = states_.find(key);
        if (found == states_.end() || !(found->second == state)) invalid("state is not live");
        auto found_release = releasing_.find(key);
        if (found_release != releasing_.end()) return found_release->second;
        auto id = "release-" + std::to_string(++release_sequence_);
        releasing_.emplace(key, id);
        return id;
    }

    void confirm_release(std::uint64_t stage_id, const StateHandle& state, const std::string& operation_id) {
        std::lock_guard<std::mutex> guard(lock_);
        const auto key = key_for(stage_id, state);
        const auto operation = releasing_.find(key);
        const auto live = states_.find(key);
        if (operation == releasing_.end() || operation->second != operation_id
                || live == states_.end() || !(live->second == state))
            invalid("release acknowledgement identity mismatch");
        states_.erase(key); retired_.erase(key); releasing_.erase(key);
    }

    std::uint64_t outstanding_bytes() const {
        std::lock_guard<std::mutex> guard(lock_);
        return used_bytes_;
    }

private:
    using StateKey = std::tuple<std::uint64_t, std::string, std::string, std::string, std::string>;
    using RequestKey = std::tuple<std::string, std::uint64_t, std::uint64_t, std::string>;
    static StateKey key_for(std::uint64_t stage_id, const StateHandle& state) {
        return {stage_id, state.backend, state.backend_instance_id, state.worker_generation, state.state_id};
    }
    [[noreturn]] static void invalid(const char* message) { throw std::invalid_argument(message); }
    bool same_request(const StageRequest& request) const {
        return active_ && request.request_id == active_->request_id && request.stage_id == active_->stage_id
            && request.epoch == active_->epoch && request.worker_generation == active_->worker_generation;
    }
    void require_request(const StageRequest& request) const {
        if (!same_request(request)) invalid("operation belongs to another request");
    }
    void validate_state(const StageRequest& request, const StateHandle& state) const {
        const auto& backend = backends_.at(request.stage_id);
        if (state.state_id.empty() || state.backend != backend.backend
                || state.backend_instance_id != backend.instance_id
                || state.worker_generation != request.worker_generation || state.epoch != request.epoch
                || state.session_id != request.session_id || state.artifact_id != request.artifact_id
                || state.layout_version != request.state_layout_version)
            invalid("state artifact, layout, session or backend identity mismatch");
        const auto key = key_for(request.stage_id, state);
        const auto found = states_.find(key);
        if (retired_.count(key) || (found != states_.end() && !(found->second == state))
                || (active_ && same_request(request) && active_->state && !(*active_->state == state)))
            invalid("backend returned a retired or different opaque state");
    }
    void retire_active_state() {
        if (active_->state) retired_.insert(key_for(active_->stage_id, *active_->state));
    }
    void release_credit(const std::string& token) {
        const auto found = pending_.find(token);
        if (found != pending_.end()) { used_bytes_ -= found->second; pending_.erase(found); }
        delivered_.erase(token);
    }

    std::map<std::uint64_t, BackendIdentity> backends_;
    std::uint64_t max_events_, max_bytes_;
    mutable std::mutex lock_;
    std::optional<StageRequest> active_;
    std::deque<StageEvent> queue_;
    std::map<std::string, std::uint64_t> pending_;
    std::set<std::string> delivered_, seen_tokens_;
    std::set<RequestKey> seen_requests_;
    std::map<StateKey, StateHandle> states_;
    std::set<StateKey> retired_;
    std::map<StateKey, std::string> releasing_;
    std::uint64_t next_seq_ = 1, last_watermark_ = 0, used_bytes_ = 0, release_sequence_ = 0;
    bool terminal_ = false, cancelled_ = false, producer_done_ = false;
};

}  // namespace omni
