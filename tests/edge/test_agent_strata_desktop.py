"""A native route's loaded configuration must not masquerade as all-op placement."""

from __future__ import annotations

import pytest

from vllm_omni.edge.agent import desktop


def _route() -> dict:
    return {
        "model": "test-quantized-model",
        "route_id": "strata-test",
        "backend": "external.strata.text.v1",
        "actual_placement": None,
        "verified_execution_configuration": "cpu+cuda:0",
        "placement_evidence_level": "native_loaded_configuration",
        "experimental": True,
        "execution_configuration_evidence": {
            "status": "verified",
            "scope": "loaded_backend_execution_configuration_not_per_request_compute",
            "private_diagnostic": "do-not-display-internal-record",
        },
    }


def _metrics() -> dict:
    return {
        "ttft_s": 1.25,
        "metrics": {
            "whole_model_wall_s": 2.5,
            "backend_metrics": {
                "private_diagnostic": "do-not-display-internal-record",
                "runtime_telemetry": {
                    "native_compute": {
                        "scope": "routed_decode_experts_only",
                        "units": ["cpu", "cuda:0"],
                        "cpu_expert_entries": 8,
                        "gpu_expert_entries": 4,
                    }
                },
            },
        },
    }


def test_route_summary_distinguishes_loaded_configuration_and_unverified_compute():
    route = _route()
    assert desktop._placement_text(route) == ("Loaded: CPU + GPU 0. Complete compute placement is not verified.")
    route["execution_configuration_evidence"]["status"] = "unverified"
    assert desktop._placement_text(route) == "Placement: not verified"
    route = _route()
    route["backend"] = "external.llamacpp.text.v1"
    assert desktop._placement_text(route) == "Placement: not verified"
    route["actual_placement"] = "cpu"
    assert desktop._placement_text(route) == "Placement: cpu"
    route["actual_placement"] = None
    route["placement_evidence_level"] = "override_selection_only"
    assert "final storage/compute unverified" in desktop._placement_text(route)


def test_metrics_summary_keeps_execution_scope_and_omits_internal_records():
    metrics = _metrics()
    text = desktop._model_metric_summary(metrics)
    assert "Model response: 2.50s" in text and "First visible text: 1.25s" in text
    assert "Decode experts executed on CPU + GPU 0; other operators not verified" in text
    assert "do-not-display" not in text
    native = metrics["metrics"]["backend_metrics"]["runtime_telemetry"]["native_compute"]
    native["scope"] = "unknown"
    assert "executed" not in desktop._model_metric_summary(metrics)
    native["scope"] = "routed_decode_experts_only"
    native["units"] = None
    assert "executed" not in desktop._model_metric_summary(metrics)
    assert "unavailable" in desktop._model_metric_summary({"ttft_s": float("nan")})


def test_ordered_native_window_preserves_answer_and_shows_scoped_route(monkeypatch):
    if desktop.QApplication is None:
        pytest.skip("PySide6 is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = desktop.QApplication.instance() or desktop.QApplication([])

    class Controller:
        def add_listener(self, callback):
            self.callback = callback

    controller = Controller()
    window = desktop.AgentWindow(controller)

    def event(seq, kind, payload):
        return {"request_id": "real-route-shape", "epoch": 1, "seq": seq, "kind": kind, "payload": payload}

    try:
        window._start_request("Say READY")
        window._accept_event(event(3, "text_delta", {"text": "READY"}))
        assert "Loaded:" not in window.placement_label.text()
        window._accept_event(event(1, "user_observation", {"text": "Say READY"}))
        window._accept_event(event(2, "route", _route()))
        window._accept_event(event(3, "text_delta", {"text": "duplicate-must-not-appear"}))
        window._accept_event(event(4, "model_metrics", _metrics()))
        window._accept_event(event(5, "final", {"answer": "READY", "streamed": True}))
        assert window.send_button.isEnabled() and not window.cancel_button.isEnabled()
        assert "Complete compute placement is not verified" in window.placement_label.text()
        assert "EXPERIMENTAL" in window.route_label.text()
        assert "strata-test" not in window.route_label.text()
        assert "strata-test" in window.route_label.toolTip()
        assert window.status_label.text() == "Ready"
        transcript = window.transcript.toPlainText()
        assert transcript.count("READY") == 2  # Original user prompt plus one streamed answer.
        assert "duplicate-must-not-appear" not in transcript
        assert "[user_observation]" not in transcript
        assert "do-not-display-internal-record" not in transcript
        assert "Model response: 2.50s" in transcript
        assert "other operators not verified" in transcript
        assert "\nAssistant: READY" in transcript
    finally:
        controller.callback = None
        window.setAttribute(desktop.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.close()
        app.processEvents()
