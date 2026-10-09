"""Synthetic protocol fixtures only. Prepared source; NOT EXECUTED yet.

These records are not compiled-header output, model evidence or qualification.
The ABI geometry and owner/runtime digests below are deliberately synthetic.
"""

import copy
import json
import unittest

from vllm_omni.engine.backends.strata_execution import strata_exec as sut

TOKEN = "a" * 64
LAYOUT = {
    "state_bytes": 8000, "frame_buffer_bytes": 32768, "frame_state_bytes": 32784,
    "ticket_bytes": 16, "concurrent_ticket_count_verified": False, "root_registry_capacity": 16,
    "overhead_scope": sut.OVERHEAD_SCOPE,
}


def runtime_binding():
    return {
        "schema": sut.RUNTIME_SCHEMA,
        "authorization": "externally_reviewed_combined_runtime_not_established_by_parser",
        "native_schema": sut.NATIVE_SCHEMA, "header_sha256": sut.HEADER_SHA256,
        "header_v1_sha256": sut.HEADER_V1_SHA256, "schema_sha256": sut.SCHEMA_SHA256,
        "incremental_patch_sha256": sut.INCREMENTAL_PATCH_SHA256,
        "boundary_patch_sha256": sut.BOUNDARY_PATCH_SHA256,
        "base_io_patch_sha256": sut.BASE_IO_PATCH_SHA256,
        "combined_patch_manifest_sha256": sut.COMBINED_PATCH_MANIFEST_SHA256,
        "runtime_manifest_sha256": "b" * 64, "native_executable_sha256": "c" * 64,
        "owner_adapter_sha256": "d" * 64, "loaded_module_receipt_sha256": "e" * 64,
        "verification_receipt_sha256": "f" * 64, "observer_layout": copy.deepcopy(LAYOUT),
    }


def owner_binding():
    return {
        "schema": sut.OWNER_SCHEMA, "worker_generation": "synthetic-generation-1",
        "pid": 1234, "creation_filetime_100ns": 134359400000000001, "stage_id": 0,
        "gpu": {"uuid": "GPU-11111111-2222-3333-4444-555555555555",
                "pci_bus_id": "00000000:01:00.0", "name_sha256": "1" * 64},
    }


def snapshot(seq=1, index=0, *, count=None, issues=0, peak=64, cancelled=False):
    count = seq * 10 + index if count is None else count
    cpu, row_phases, cuda = [], [], []
    for phase in sut.PHASES:
        for family in sut.CPU_FAMILIES:
            cpu.append({"phase": phase, "family": family, "unit": "CPU_expert_method",
                        **{key: count for key in sut.CPU_COUNTS}})
        for mode in range(1, 7):
            row_phases.append({"phase": phase, "mode": mode, "unit": "CPU_row_partition_phase",
                               **{key: count for key in sut.CPU_PHASE_COUNTS}})
        for family in sut.CUDA_FAMILIES:
            cuda.append({"phase": phase, "family": family,
                         "unit": "CUDA_graph_replay_including_host_copy_nodes",
                         **{key: (0 if key in ("launch_errors", "fence_errors") else count) for key in sut.CUDA_COUNTS},
                         "last_launch_status": None, "last_fence_status": 0})
    return {
        "schema": sut.NATIVE_SCHEMA, "native_request_seq": seq, "snapshot_seq": index,
        "phase": sut.BOUNDARIES[index],
        "clock": {"source": "Windows_QPC", "ticks": seq * 1000 + index, "frequency_hz": 10_000_000},
        "counter_scope": "process_lifetime_cumulative_phase_at_submission", "snapshot_coherent": False,
        "scope_status": "partial_coverage",
        "request_terminal": ("cancelled" if cancelled else "completed") if index == 2 else None,
        "boundary_complete": index == 2 and not cancelled and issues == 0, "issues": issues,
        "observer": copy.deepcopy(LAYOUT), "inflight": {"cpu_methods_and_phases": 0, "cuda_replays": 0},
        "counters": {"cpu": cpu, "cpu_phase": row_phases, "cuda": cuda},
        "memory": {"ordinary_expert_cache": {
            "unit": "bytes", "accounting": "successful_owned_cudaMalloc_requested_bytes",
            **{key: (0 if key in ("allocation_errors", "failed_frees") else count) for key in sut.MEMORY_COUNTS}, "last_allocation_status": 0, "last_free_status": None,
            "tracked_live_requested_bytes": peak, "lifetime_peak_requested_bytes": 64,
            "request_peak_requested_bytes": peak, "untracked_cache_family_bits": 0,
            "physical_resident_bytes": None, "vmm_reserved_bytes": None, "vmm_mapped_committed_bytes": None,
        }},
        "coverage": {"instrumented": list(sut.INSTRUMENTED), "excluded": list(sut.EXCLUDED), "entire_model_covered": False},
        "whole_model_placement": None, "physical_ssd_read_bytes": None,
        "aggregate_gpu_hard_cap_verified": False, "aggregate_ram_hard_cap_verified": False,
    }


def native_line(record):
    return sut.PREFIX + json.dumps(record, separators=(",", ":")) + "\n"


class Harness:
    def __init__(self):
        self.owner = owner_binding()
        self.observer = sut.StrataExecutionObserver(runtime_binding(), self.owner, channel_token=TOKEN)
        self.request = "request-1"
        self.epoch = 1
        self.observer.begin(self.request, self.epoch)

    def frame(self, kind, payload):
        return {
            "schema": sut.FRAME_SCHEMA, "channel_token": TOKEN,
            **self.observer.binding_identity,
            "worker_generation": self.owner["worker_generation"], "pid": self.owner["pid"],
            "creation_filetime_100ns": self.owner["creation_filetime_100ns"],
            "gpu_uuid": self.owner["gpu"]["uuid"], "stage_id": self.owner["stage_id"],
            "request_id": self.request, "epoch": self.epoch, "kind": kind, "payload": payload,
        }

    def send(self, kind, payload):
        self.observer.ingest(self.frame(kind, payload))

    def dispatch(self, seq=1):
        self.send("dispatch", {"command": "GEN", "native_request_seq": seq})

    def complete(self, seq=1, peak=64):
        self.dispatch(seq)
        for index in range(3):
            self.send("snapshot", snapshot(seq, index, peak=peak))
        self.send("native_done", {"native_request_seq": seq, "finish_reason": "stop"})
        return self.observer.finish(self.request, self.epoch, omni_completed=True)


class Parsing(unittest.TestCase):
    def test_valid_native_line_is_detached_untrusted_data(self):
        data = snapshot()
        parsed = sut.parse_native_line(native_line(data))
        self.assertEqual(parsed, data)
        self.assertIsNot(parsed, data)
        self.assertEqual(sut.parse_native_line(native_line(data).replace("\n", "\r\n")), data)

    def test_duplicate_nonfinite_and_float_numbers_refuse(self):
        valid = native_line(snapshot())
        mutations = (
            valid.replace('"snapshot_seq":0', '"snapshot_seq":0,"snapshot_seq":0'),
            valid.replace('"snapshot_seq":0', '"snapshot_seq":NaN'),
            valid.replace('"snapshot_seq":0', '"snapshot_seq":Infinity'),
            valid.replace('"snapshot_seq":0', '"snapshot_seq":1e999'),
            valid.replace('"snapshot_seq":0', '"snapshot_seq":0.0'),
        )
        for line in mutations:
            with self.subTest(kind="malformed_number_or_duplicate"), self.assertRaises(sut.ObservationError):
                sut.parse_native_line(line)

    def test_exact_native_keys_literals_bounds_and_units(self):
        variants = []
        for key, value in (
            ("native_request_seq", True), ("native_request_seq", 1 << 62), ("snapshot_seq", False),
            ("issues", 1 << 14), ("whole_model_placement", "CUDA"), ("physical_ssd_read_bytes", 0),
            ("aggregate_gpu_hard_cap_verified", True), ("snapshot_coherent", True), ("phase", "prefill"),
        ):
            data = snapshot()
            data[key] = value
            variants.append(data)
        data = snapshot()
        data["unexpected"] = 0
        variants.append(data)
        data = snapshot()
        data["counters"]["cpu"][0]["unit"] = "kernel"
        variants.append(data)
        data = snapshot()
        data["counters"]["cpu_phase"][0]["mode"] = True
        variants.append(data)
        data = snapshot()
        data["counters"]["cpu"][0]["submitted"] = -1
        variants.append(data)
        data = snapshot()
        data["memory"]["ordinary_expert_cache"]["untracked_cache_family_bits"] = 4
        variants.append(data)
        for data in variants:
            with self.subTest(kind="shape_bound_or_literal"), self.assertRaises(sut.ObservationError):
                sut.validate_snapshot(data)

    def test_array_order_and_coverage_are_source_bound(self):
        for mutate in (
            lambda data: data["counters"]["cpu"].reverse(),
            lambda data: data["counters"]["cuda"].append(data["counters"]["cuda"][0]),
            lambda data: data["coverage"]["excluded"].pop(),
        ):
            data = snapshot()
            mutate(data)
            with self.assertRaises(sut.ObservationError):
                sut.validate_snapshot(data)

    def test_signed_status_is_not_a_counter_or_success_assertion(self):
        for value in (None, 0, 1, -(1 << 31), (1 << 31) - 1):
            data = snapshot()
            data["counters"]["cuda"][0]["last_launch_status"] = value
            sut.validate_snapshot(data)
        for value in (-1, True, 0.0, 1 << 31):
            data = snapshot()
            data["counters"]["cuda"][0]["last_launch_status"] = value
            with self.assertRaises(sut.ObservationError):
                sut.validate_snapshot(data)

    def test_prefix_encoding_frame_size_and_multiline_refuse(self):
        for line in ("native io " + native_line(snapshot()), native_line(snapshot()).rstrip("\n"),
                     native_line(snapshot()) + "\n", "OMNI_EXEC_V1 " + " " * 32768 + "\n"):
            with self.assertRaises(sut.ObservationError):
                sut.parse_native_line(line)

    def test_unavailable_qpc_must_have_explicit_issue_and_null_pair(self):
        data = snapshot(issues=sut.CLOCK_UNAVAILABLE)
        data["clock"]["ticks"] = None
        data["clock"]["frequency_hz"] = None
        sut.validate_snapshot(data)
        for mutate in (
            lambda value: value.update(issues=0),
            lambda value: value["clock"].update(frequency_hz=1_000_000_000),
        ):
            invalid = copy.deepcopy(data)
            mutate(invalid)
            with self.assertRaises(sut.ObservationError):
                sut.validate_snapshot(invalid)


class OwnershipAndLifecycle(unittest.TestCase):
    def test_complete_has_only_partial_coverage_and_no_qualification(self):
        report = Harness().complete()
        self.assertTrue(report["complete"])
        self.assertFalse(report["binding_verified_by_parser"])
        self.assertFalse(report["coverage"]["entire_model_covered"])
        self.assertFalse(report["runtime_qualification"])
        self.assertIsNone(report["whole_model_placement"])
        self.assertIsNone(report["physical_ssd_read_bytes"])
        self.assertEqual(len(report["snapshots"]), 3)
        self.assertEqual(len(report["counter_deltas"]), 2)
        self.assertLessEqual(len(json.dumps(report).encode()), sut.REPORT_BYTES)

    def test_legacy_runtime_and_wrong_source_chain_refuse(self):
        for key, value in (("schema", "omni-strata-observation-runtime-v1"),
                           ("header_sha256", sut.HEADER_V1_SHA256),
                           ("combined_patch_manifest_sha256", "0" * 64)):
            runtime = runtime_binding()
            runtime[key] = value
            with self.assertRaises(sut.ObservationError):
                sut.StrataExecutionObserver(runtime, owner_binding(), channel_token=TOKEN)

    def test_no_bare_native_record_can_establish_owner(self):
        harness = Harness()
        with self.assertRaises(sut.ObservationError):
            harness.observer.ingest(snapshot())
        self.assertTrue(harness.observer.retired)

    def test_owner_epoch_gpu_runtime_and_channel_mismatch_retire(self):
        for key, value in (
            ("pid", 1235), ("pid", 1234.0), ("creation_filetime_100ns", 134359400000000002),
            ("worker_generation", "other-generation"), ("epoch", True), ("epoch", 2),
            ("stage_id", False), ("request_id", "other-request"), ("gpu_uuid", "GPU-other"),
            ("runtime_binding_sha256", "0" * 64), ("owner_binding_sha256", "0" * 64),
            ("channel_token", "0" * 64), ("channel_token", "\u00e9" * 64),
        ):
            harness = Harness()
            frame = harness.frame("dispatch", {"command": "GEN", "native_request_seq": 1})
            frame[key] = value
            with self.subTest(field=key), self.assertRaises(sut.ObservationError):
                harness.observer.ingest(frame)
            self.assertTrue(harness.observer.retired)

    def test_exact_compiled_geometry_not_synthetic_size_inference(self):
        harness = Harness()
        harness.dispatch()
        data = snapshot()
        data["observer"]["ticket_bytes"] += 1
        with self.assertRaises(sut.ObservationError):
            harness.send("snapshot", data)

    def test_missing_native_end_remains_partial_after_omni_completion(self):
        harness = Harness()
        harness.dispatch()
        harness.send("snapshot", snapshot())
        harness.send("native_done", {"native_request_seq": 1, "finish_reason": "stop"})
        report = harness.observer.finish(harness.request, harness.epoch, omni_completed=True)
        self.assertFalse(report["complete"])
        self.assertIn("missing_native_snapshots", report["reasons"])
        self.assertTrue(harness.observer.retired)

    def test_cancelled_native_terminal_is_not_cleaned_by_drain(self):
        harness = Harness()
        harness.dispatch()
        for index in range(3):
            harness.send("snapshot", snapshot(index=index, cancelled=True))
        harness.send("native_done", {"native_request_seq": 1, "finish_reason": "cancelled"})
        report = harness.observer.finish(harness.request, harness.epoch, omni_completed=False, cancellation_requested=True)
        self.assertFalse(report["complete"])
        self.assertTrue(harness.observer.retired)

    def test_later_issue_or_inflight_disagreement_is_partial_not_parse_error(self):
        for field in ("issues", "inflight", "error_counter", "error_status", "memory_error"):
            harness = Harness()
            harness.dispatch()
            for index in range(3):
                data = snapshot(index=index)
                if index == 2:
                    data["boundary_complete"] = True
                    if field == "issues":
                        data["issues"] = 256
                    elif field == "inflight":
                        data["inflight"]["cuda_replays"] = 1
                    elif field == "error_counter":
                        data["counters"]["cuda"][0]["launch_errors"] = 1
                    elif field == "error_status":
                        data["counters"]["cuda"][0]["last_fence_status"] = 700
                    else:
                        data["memory"]["ordinary_expert_cache"]["failed_frees"] = 1
                harness.send("snapshot", data)
            harness.send("native_done", {"native_request_seq": 1, "finish_reason": "stop"})
            report = harness.observer.finish(harness.request, harness.epoch, omni_completed=True)
            self.assertFalse(report["complete"])
            self.assertTrue(harness.observer.retired)

    def test_independent_loads_do_not_impose_instantaneous_inequalities(self):
        harness = Harness()
        harness.dispatch()
        for index in range(3):
            data = snapshot(index=index)
            data["counters"]["cpu"][0]["submitted"] = 1
            data["counters"]["cpu"][0]["completed"] = 99
            cache = data["memory"]["ordinary_expert_cache"]
            cache["tracked_live_requested_bytes"] = 1000 - index
            cache["lifetime_peak_requested_bytes"] = 1
            cache["request_peak_requested_bytes"] = 1
            harness.send("snapshot", data)
        harness.send("native_done", {"native_request_seq": 1, "finish_reason": "stop"})
        self.assertTrue(harness.observer.finish(harness.request, harness.epoch, omni_completed=True)["complete"])

    def test_per_counter_regression_and_sticky_mask_loss_refuse(self):
        for kind in ("counter", "lifetime_peak", "issue", "cache_bits", "request_peak"):
            harness = Harness()
            harness.dispatch()
            first, next_record = snapshot(index=0), snapshot(index=1)
            if kind == "counter":
                next_record["counters"]["cuda"][0]["submitted"] = 0
            elif kind == "lifetime_peak":
                next_record["memory"]["ordinary_expert_cache"]["lifetime_peak_requested_bytes"] = 0
            elif kind == "issue":
                first["issues"] = 1
            elif kind == "cache_bits":
                first["memory"]["ordinary_expert_cache"]["untracked_cache_family_bits"] = 1
            else:
                next_record["memory"]["ordinary_expert_cache"]["request_peak_requested_bytes"] = 0
            harness.send("snapshot", first)
            with self.subTest(kind=kind), self.assertRaises(sut.ObservationError):
                harness.send("snapshot", next_record)

    def test_peak_can_reset_between_requests_but_lifetime_counts_cannot(self):
        harness = Harness()
        self.assertTrue(harness.complete(peak=64)["complete"])
        harness.request, harness.epoch = "request-2", 2
        harness.observer.begin(harness.request, harness.epoch)
        self.assertTrue(harness.complete(seq=2, peak=16)["complete"])

    def test_snapshot_clock_sequence_done_order_and_epoch_are_strict(self):
        for kind in ("snapshot_order", "native_seq", "clock", "frequency", "done", "duplicate_dispatch"):
            harness = Harness()
            harness.dispatch()
            harness.send("snapshot", snapshot())
            data = snapshot(index=1)
            if kind == "snapshot_order":
                data = snapshot(index=2)
            elif kind == "native_seq":
                data["native_request_seq"] = 2
            elif kind == "clock":
                data["clock"]["ticks"] = 1
            elif kind == "frequency":
                data["clock"]["frequency_hz"] += 1
            with self.subTest(kind=kind), self.assertRaises(sut.ObservationError):
                if kind == "done":
                    harness.send("native_done", {"native_request_seq": 2, "finish_reason": "stop"})
                elif kind == "duplicate_dispatch":
                    harness.dispatch()
                else:
                    harness.send("snapshot", data)

    def test_stale_next_request_dispatch_and_reused_epoch_refuse(self):
        harness = Harness()
        harness.complete()
        harness.request, harness.epoch = "request-2", 2
        harness.observer.begin(harness.request, harness.epoch)
        with self.assertRaises(sut.ObservationError):
            harness.dispatch(seq=1)
        second = Harness()
        second.complete()
        with self.assertRaises(sut.ObservationError):
            second.observer.begin("request-again", 1)

    def test_null_native_clock_is_partial_not_fabricated_one_ghz(self):
        harness = Harness()
        harness.dispatch()
        for index in range(3):
            data = snapshot(index=index, issues=sut.CLOCK_UNAVAILABLE)
            data["clock"]["ticks"] = data["clock"]["frequency_hz"] = None
            harness.send("snapshot", data)
        harness.send("native_done", {"native_request_seq": 1, "finish_reason": "stop"})
        report = harness.observer.finish(harness.request, harness.epoch, omni_completed=True)
        self.assertFalse(report["complete"])
        self.assertIsNone(report["snapshots"][2]["clock"]["frequency_hz"])

    def test_output_and_last_report_do_not_mutate_accumulator_state(self):
        harness = Harness()
        report = harness.complete()
        report["owner_binding"]["pid"] = 0
        report["snapshots"][0]["counters"]["cpu"][0]["submitted"] = 0
        retained = harness.observer.last_observation()
        self.assertEqual(retained["owner_binding"]["pid"], 1234)
        retained["owner_binding"]["pid"] = 0
        self.assertEqual(harness.observer.last_observation()["owner_binding"]["pid"], 1234)
        self.assertNotIn(TOKEN, json.dumps(harness.observer.last_observation()))


if __name__ == "__main__":
    unittest.main()
