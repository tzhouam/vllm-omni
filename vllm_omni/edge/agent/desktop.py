"""Optional PySide6 desktop shell for the local Omni edge agent.

The controller owns inference, memory and the tool policy.  This window only
subscribes to its ordered events and presents UI-only approval callbacks.
Importing this module on Linux or without PySide6 is safe; constructing the
window then reports the missing optional dependency plainly.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import hashlib
import json
import math
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import parse_qsl

from vllm_omni.edge.agent.tools import _SCREEN_CAPTURE_SCOPE, _foreground_window_target

try:
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QApplication = None  # type: ignore[assignment,misc]
    QMainWindow = object  # type: ignore[assignment,misc]


class DesktopController(Protocol):
    """UI adapter; ``approve`` must be absent from model-facing tools."""

    def add_listener(self, callback: Callable[[Any], None]) -> None: ...
    def submit(self, prompt: str) -> Any: ...
    def submit_read_url(self, url: str, instruction: str) -> Any: ...
    def cancel(self) -> Any: ...
    def approve(self, challenge_id: str) -> Any: ...
    def reject(self, challenge_id: str) -> Any: ...
    def delete_all_memory(self) -> int: ...


def _field(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _display_payload(value: Any) -> Any:
    """Keep large image payloads out of the transcript; memory sees originals."""
    if isinstance(value, Mapping):
        return {
            key: f"<image data, {value.get('size_bytes', 'unknown')} bytes>" if key == "base64"
            else "<POST body; inspect the exact approval review>" if key == "body_b64"
            else _display_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_display_payload(item) for item in value]
    return value


def _placement_text(payload: Mapping[str, Any]) -> str:
    """Present a loaded configuration separately from observed computation."""
    observed = payload.get("actual_placement")
    evidence = payload.get("execution_configuration_evidence")
    if (
        observed is None
        and payload.get("backend") == "external.strata.text.v1"
        and payload.get("placement_evidence_level") == "native_loaded_configuration"
        and isinstance(evidence, Mapping)
        and evidence.get("status") == "verified"
        and evidence.get("scope") == "loaded_backend_execution_configuration_not_per_request_compute"
        and payload.get("verified_execution_configuration") == "cpu+cuda:0"
    ):
        return "Loaded: CPU + GPU 0. Complete compute placement is not verified."
    placement = "Placement: " + (str(observed) if observed is not None else "not verified")
    if payload.get("placement_evidence_level") == "override_selection_only":
        placement += " — expert override selected; final storage/compute unverified"
    return placement


def _model_metric_summary(payload: Mapping[str, Any]) -> str:
    """Show decision-useful timings without dumping internal stage records."""
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    parts = []
    for label, value in (("Model response", metrics.get("whole_model_wall_s")),
                         ("First visible text", payload.get("ttft_s"))):
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            parts.append(f"{label}: {value:.2f}s")
    backend = metrics.get("backend_metrics")
    backend = backend if isinstance(backend, Mapping) else {}
    telemetry = backend.get("runtime_telemetry")
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    native = telemetry.get("native_compute")
    if isinstance(native, Mapping) and native.get("scope") == "routed_decode_experts_only":
        observed_units = native.get("units")
        observed_units = observed_units if isinstance(observed_units, (list, tuple)) else ()
        units = []
        for key, unit, label in (("cpu_expert_entries", "cpu", "CPU"),
                                 ("gpu_expert_entries", "cuda:0", "GPU 0")):
            count = native.get(key)
            if type(count) is int and count > 0 and unit in observed_units:
                units.append(label)
        if units:
            parts.append("Decode experts executed on " + " + ".join(units) + "; other operators not verified")
    return "; ".join(parts) if parts else "Model request finished; timing unavailable"


def _browser_post_review(challenge: Mapping[str, Any]) -> str:
    """Render every approved POST byte while checking the bound UI metadata."""
    action = challenge.get("action")
    action = action if isinstance(action, Mapping) else {}
    operation = challenge.get("operation", action.get("operation"))
    arguments = challenge.get("arguments", action.get("arguments"))
    target = challenge.get("target")
    if operation != "browser_post" or not isinstance(arguments, Mapping) or not isinstance(target, Mapping):
        raise ValueError("POST approval has no exact action and target")
    url = arguments.get("url")
    content_type = arguments.get("content_type")
    body_b64 = arguments.get("body_b64")
    if (not isinstance(url, str) or not url or not isinstance(content_type, str)
            or not isinstance(body_b64, str)):
        raise ValueError("POST approval has incomplete request data")
    try:
        body = base64.b64decode(body_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("POST approval body is not canonical base64") from exc
    if base64.b64encode(body).decode("ascii") != body_b64:
        raise ValueError("POST approval body is not canonical base64")
    digest = hashlib.sha256(body).hexdigest()
    if (target.get("url") != url or target.get("content_type") != content_type
            or type(target.get("body_size")) is not int or target["body_size"] != len(body)
            or target.get("body_sha256") != digest):
        raise ValueError("POST approval target, media type, size or SHA-256 does not match its body")
    fingerprint = target.get("cookie_fingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)):
        raise ValueError("POST approval has no cookie context fingerprint")
    lines = [
        "HTTP POST — review the exact request before approving",
        f"Target URL: {url}",
        f"Current page: {target.get('current_url', 'unreported')}",
        f"Content-Type: {content_type}",
        f"Body size: {len(body)} bytes",
        f"Body SHA-256: {digest}",
        f"Cookie context fingerprint: {fingerprint}",
        f"Applicable cookies: {target.get('cookie_count', 'unreported')}",
        "Redirects and automatic retries: disabled",
    ]
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        lines.extend(["", "Body is not UTF-8; complete bytes appear below."])
    else:
        lines.extend(["", "Complete UTF-8 body (original whitespace preserved):", decoded])
        if content_type == "application/x-www-form-urlencoded":
            fields = parse_qsl(decoded, keep_blank_values=True)
            lines.extend([
                "", "Decoded form fields (order and duplicate names preserved):",
                json.dumps(fields, ensure_ascii=False, indent=2),
            ])
        elif content_type == "application/json":
            try:
                parsed = json.loads(decoded)
            except json.JSONDecodeError:
                pass
            else:
                lines.extend([
                    "", "Parsed JSON for review (the original bytes above remain authoritative):",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                ])
    # A bytewise representation remains available even for text with hidden
    # controls, CRLF normalization, or Unicode direction marks in a Qt widget.
    lines.extend(["", "Complete request body bytes (hex):", body.hex(" ")])
    return "\n".join(lines)


def _screen_capture_review(challenge: Mapping[str, Any]) -> str:
    """Show the exact foreground-window scope before a sensitive read."""
    action = challenge.get("action")
    action = action if isinstance(action, Mapping) else {}
    operation = challenge.get("operation", action.get("operation"))
    target = challenge.get("target")
    if (operation != "screen_capture" or challenge.get("risk") != "sensitive_desktop_read"
            or not isinstance(target, Mapping)
            or target.get("capture_scope") != _SCREEN_CAPTURE_SCOPE):
        raise ValueError("screen capture approval has no verified foreground scope")
    lines = [
        "Sensitive desktop read — one screen capture within foreground-window bounds",
        "Scope: visible screen pixels inside the current foreground window's bounding rectangle.",
        "Overlays or other windows can appear; desktop background may show through transparent corners.",
        "A maximized foreground window may make this rectangle cover the whole screen.",
    ]
    if (type(target.get("window_handle")) is not int
            or type(target.get("process_id")) is not int
            or not isinstance(target.get("window_title"), str)
            or not isinstance(target.get("source_bbox"), Mapping)):
        raise ValueError("screen capture approval has incomplete window identity")
    lines.extend([
        "Window title: " + json.dumps(target["window_title"] or "<untitled>", ensure_ascii=False),
        f"Process ID: {target['process_id']}",
        f"Window handle: {target['window_handle']}",
        f"Rectangle: {json.dumps(dict(target['source_bbox']), ensure_ascii=False)}",
        "If the foreground window changes before or during capture, this approval fails.",
    ])
    return "\n".join(lines)


def _restore_capture_foreground(target: Mapping[str, Any]) -> None:
    """Hand focus back after the review dialog, before dispatching approval."""
    if sys.platform != "win32" or type(target.get("window_handle")) is not int:
        raise RuntimeError("approved foreground window cannot be restored")
    user32 = ctypes.windll.user32
    user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    user32.SetForegroundWindow.restype = ctypes.c_bool
    if not user32.SetForegroundWindow(ctypes.c_void_p(target["window_handle"])):
        raise RuntimeError("Windows did not restore the approved foreground window")
    if dict(_foreground_window_target()) != dict(target):
        raise RuntimeError("foreground window changed after desktop approval")


if QApplication is None:

    class AgentWindow:  # type: ignore[no-redef]
        def __init__(self, controller: DesktopController) -> None:
            del controller
            raise RuntimeError("PySide6 is required for the native Windows Agent UI")

else:

    class _Relay(QObject):
        event_received = Signal(object)
        callback_error = Signal(str)
        memory_cleared = Signal(int)
        approval_resolved = Signal(str)
        approval_error = Signal(str, str, str)


    class AgentWindow(QMainWindow):  # type: ignore[no-redef]
        """Single-request window with sequence fencing and explicit approvals."""

        _MAX_REORDER = 256

        def __init__(self, controller: DesktopController) -> None:
            super().__init__()
            self.controller = controller
            self.setWindowTitle("Omni Edge Agent")
            self.resize(1040, 760)
            self._relay = _Relay()
            self._relay.event_received.connect(self._accept_event)
            self._relay.callback_error.connect(self._report_callback_error)
            self._relay.memory_cleared.connect(self._memory_cleared)
            self._relay.approval_resolved.connect(self._approval_resolved)
            self._relay.approval_error.connect(self._report_approval_error)
            self._last_seq: dict[tuple[str, int], int] = {}
            self._pending: dict[tuple[str, int], dict[int, Any]] = {}
            self._max_epoch: dict[str, int] = {}
            self._active = False
            self._answer_stream_key: tuple[str, int] | None = None

            root = QWidget(self)
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            self.route_label = QLabel("Route: waiting for a qualified local plan")
            self.route_label.setWordWrap(True)
            self.placement_label = QLabel("Placement: not loaded")
            self.placement_label.setWordWrap(True)
            self.status_label = QLabel("Ready")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.route_label)
            layout.addWidget(self.placement_label)
            layout.addWidget(self.status_label)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            self.transcript = QPlainTextEdit()
            self.transcript.setReadOnly(True)
            self.transcript.setPlaceholderText("Agent events and answers appear here")
            splitter.addWidget(self.transcript)

            approval_panel = QWidget()
            approval_layout = QVBoxLayout(approval_panel)
            approval_layout.addWidget(QLabel("Actions requiring your approval"))
            self.approvals = QListWidget()
            approval_layout.addWidget(self.approvals)
            approval_buttons = QHBoxLayout()
            self.approve_button = QPushButton("批准 / Approve")
            self.reject_button = QPushButton("拒绝 / Reject")
            self.approve_button.clicked.connect(self._approve_selected)
            self.reject_button.clicked.connect(self._reject_selected)
            approval_buttons.addWidget(self.approve_button)
            approval_buttons.addWidget(self.reject_button)
            approval_layout.addLayout(approval_buttons)
            splitter.addWidget(approval_panel)
            splitter.setSizes([720, 320])
            layout.addWidget(splitter, stretch=1)

            self.prompt = QPlainTextEdit()
            self.prompt.setPlaceholderText("Describe a browser, Windows Settings, memory, or code task")
            self.prompt.setMaximumHeight(110)
            layout.addWidget(self.prompt)
            read_controls = QHBoxLayout()
            read_controls.addWidget(QLabel("URL to read:"))
            self.read_url_field = QLineEdit()
            self.read_url_field.setPlaceholderText("https://example.com/page")
            self.read_url_field.setAccessibleName("URL for explicit Read URL action")
            read_controls.addWidget(self.read_url_field, stretch=1)
            self.read_url_button = QPushButton("读取 URL / Read URL")
            self.read_url_button.clicked.connect(self._submit_read_url)
            read_controls.addWidget(self.read_url_button)
            layout.addLayout(read_controls)
            controls = QHBoxLayout()
            self.send_button = QPushButton("发送 / Send")
            self.cancel_button = QPushButton("取消 / Cancel")
            self.clear_memory_button = QPushButton("清除本地记忆 / Clear memory")
            self.cancel_button.setEnabled(False)
            self.send_button.clicked.connect(self._submit)
            self.cancel_button.clicked.connect(self._cancel)
            self.clear_memory_button.clicked.connect(self._delete_all_memory)
            controls.addWidget(self.send_button)
            controls.addWidget(self.cancel_button)
            controls.addWidget(self.clear_memory_button)
            layout.addLayout(controls)
            self.controller.add_listener(self.on_event)

        def on_event(self, event: Any) -> None:
            """Thread-safe listener; Qt queues the rendering on its UI thread."""
            self._relay.event_received.emit(event)

        def _invoke(self, callback: Callable[..., Any], *args: Any) -> None:
            def run() -> None:
                try:
                    callback(*args)
                except Exception as exc:
                    self._relay.callback_error.emit(f"{type(exc).__name__}: {exc}")

            threading.Thread(target=run, daemon=True).start()

        def _submit(self) -> None:
            prompt = self.prompt.toPlainText().strip()
            if not prompt or self._active:
                return
            self._start_request(prompt)
            self._invoke(self.controller.submit, prompt)

        def _submit_read_url(self) -> None:
            url = self.read_url_field.text().strip()
            instruction = self.prompt.toPlainText().strip()
            if not url or not instruction or self._active:
                return
            self._start_request(
                f"Read URL: {url}\nInstruction: {instruction}", clear_prompt=False,
            )
            self._invoke(self.controller.submit_read_url, url, instruction)

        def _start_request(self, displayed_task: str, *, clear_prompt: bool = True) -> None:
            if clear_prompt:
                self.prompt.clear()
            self.transcript.appendPlainText(f"\nYou: {displayed_task}\n")
            self._active = True
            self.send_button.setEnabled(False)
            self.read_url_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.clear_memory_button.setEnabled(False)
            self.status_label.setText("Running one local request")

        def _cancel(self) -> None:
            if self._active:
                self.status_label.setText("Cancelling request and releasing state…")
                self._invoke(self.controller.cancel)

        def _delete_all_memory(self) -> None:
            if self._active:
                return
            choice = QMessageBox.question(
                self, "Clear local Agent memory",
                "Delete every saved observation, conversation and derived search entry?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
            self.clear_memory_button.setEnabled(False)

            def clear() -> None:
                try:
                    self._relay.memory_cleared.emit(self.controller.delete_all_memory())
                except Exception as exc:
                    self._relay.callback_error.emit(f"{type(exc).__name__}: {exc}")

            threading.Thread(target=clear, daemon=True).start()

        def _memory_cleared(self, count: int) -> None:
            self.transcript.appendPlainText(f"[memory] deleted {count} events and their search entries")
            self.status_label.setText("Local memory cleared")
            self.clear_memory_button.setEnabled(True)

        def _approval_selected(self) -> str | None:
            item = self.approvals.currentItem()
            if item is None:
                return None
            challenge_id = item.data(Qt.ItemDataRole.UserRole)
            return str(challenge_id) if challenge_id else None

        def _remove_approval(self, challenge_id: str) -> None:
            for row in range(self.approvals.count()):
                item = self.approvals.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == challenge_id:
                    self.approvals.takeItem(row)
                    return

        def _dispatch_approval(self, challenge_id: str, *, approved: bool) -> None:
            self.approve_button.setEnabled(False)
            self.reject_button.setEnabled(False)

            def run() -> None:
                try:
                    if approved:
                        self.controller.approve(challenge_id)
                    else:
                        self.controller.reject(challenge_id)
                except Exception as exc:
                    self._relay.approval_error.emit(
                        challenge_id, type(exc).__name__, str(exc),
                    )
                else:
                    self._relay.approval_resolved.emit(challenge_id)

            threading.Thread(target=run, daemon=True).start()

        def _approval_resolved(self, challenge_id: str) -> None:
            self._remove_approval(challenge_id)
            self.approve_button.setEnabled(True)
            self.reject_button.setEnabled(True)

        def _report_approval_error(self, challenge_id: str, kind: str,
                                   message: str) -> None:
            # A stale challenge is no longer actionable. Other failures leave
            # the item available for retry without ending the active request.
            if kind in {"ValueError", "TimeoutError"}:
                self._remove_approval(challenge_id)
            self.approve_button.setEnabled(True)
            self.reject_button.setEnabled(True)
            detail = f"Approval failed ({kind}): {message}"
            self.transcript.appendPlainText(f"[approval error] {detail}")
            self.status_label.setText(detail)

        def _confirm_browser_post(self, review: str) -> bool:
            dialog = QDialog(self)
            dialog.setWindowTitle("Review exact browser POST")
            dialog.resize(900, 680)
            layout = QVBoxLayout(dialog)
            explanation = QLabel(
                "This sends one HTTP POST. Review the full target and body below; "
                "a timeout may leave the server-side result uncertain."
            )
            explanation.setWordWrap(True)
            layout.addWidget(explanation)
            preview = QPlainTextEdit(dialog)
            preview.setObjectName("browser_post_exact_review")
            preview.setReadOnly(True)
            # The URL can be close to the 2048-byte admission limit. Wrap the
            # complete review so its query tail cannot disappear off-screen.
            preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            preview.setPlainText(review)
            layout.addWidget(preview)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=dialog,
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Approve exact POST")
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            return dialog.exec() == QDialog.DialogCode.Accepted

        def _confirm_screen_capture(self, review: str) -> bool:
            dialog = QDialog(self)
            dialog.setWindowTitle("Review foreground-window capture")
            dialog.resize(680, 400)
            layout = QVBoxLayout(dialog)
            preview = QPlainTextEdit(dialog)
            preview.setObjectName("screen_capture_exact_review")
            preview.setReadOnly(True)
            preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            preview.setPlainText(review)
            layout.addWidget(preview)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=dialog,
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Approve one capture")
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            return dialog.exec() == QDialog.DialogCode.Accepted

        def _approve_selected(self) -> None:
            item = self.approvals.currentItem()
            if item is not None and item.data(Qt.ItemDataRole.UserRole + 3) == "browser_post":
                review = item.data(Qt.ItemDataRole.UserRole + 2)
                if not isinstance(review, str) or not self._confirm_browser_post(review):
                    return
            elif item is not None and item.data(Qt.ItemDataRole.UserRole + 3) == "screen_capture":
                review = item.data(Qt.ItemDataRole.UserRole + 2)
                if not isinstance(review, str) or not self._confirm_screen_capture(review):
                    return
                target = item.data(Qt.ItemDataRole.UserRole + 4)
                try:
                    if not isinstance(target, Mapping):
                        raise RuntimeError("screen capture target is missing")
                    _restore_capture_foreground(target)
                except Exception as exc:
                    detail = f"Screen capture approval held: {exc}"
                    self.status_label.setText(detail)
                    self.transcript.appendPlainText(f"[approval error] {detail}")
                    return
            elif item is not None and item.data(Qt.ItemDataRole.UserRole + 1) == "high_impact_external_action":
                choice = QMessageBox.question(
                    self, "Confirm high-impact browser action",
                    "This control may send, purchase, or delete data. Approve the exact action shown?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if choice != QMessageBox.StandardButton.Yes:
                    return
            challenge_id = self._approval_selected()
            if challenge_id:
                self._dispatch_approval(challenge_id, approved=True)

        def _reject_selected(self) -> None:
            challenge_id = self._approval_selected()
            if challenge_id:
                self._dispatch_approval(challenge_id, approved=False)

        def _accept_event(self, event: Any) -> None:
            request_id = _field(event, "request_id")
            epoch = _field(event, "epoch")
            seq = _field(event, "seq")
            if not isinstance(request_id, str) or not request_id or type(epoch) is not int or type(seq) is not int:
                self.status_label.setText("Dropped event without valid request, epoch and sequence")
                return
            if epoch < self._max_epoch.get(request_id, epoch):
                return
            if epoch > self._max_epoch.get(request_id, epoch):
                for key in [key for key in self._pending if key[0] == request_id]:
                    self._pending.pop(key, None)
                    self._last_seq.pop(key, None)
            self._max_epoch[request_id] = epoch
            key = (request_id, epoch)
            last = self._last_seq.get(key, 0)
            if seq <= last:
                return
            pending = self._pending.setdefault(key, {})
            if seq in pending:
                return
            if len(pending) >= self._MAX_REORDER:
                self.status_label.setText("Event sequence gap exceeded the UI buffer; request output paused")
                return
            pending[seq] = event
            while last + 1 in pending:
                last += 1
                self._render(pending.pop(last))
            self._last_seq[key] = last

        def _append_inline(self, value: str) -> None:
            cursor = self.transcript.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(value)
            self.transcript.setTextCursor(cursor)
            self.transcript.ensureCursorVisible()

        def _render(self, event: Any) -> None:
            kind = str(_field(event, "kind", "event"))
            payload = _field(event, "payload", _field(event, "data", {}))
            if not isinstance(payload, Mapping):
                payload = {"value": payload}
            if kind == "user_observation":
                if not self._active:
                    self.transcript.appendPlainText("You: " + str(payload.get("text", "")))
            elif kind in {"text_delta", "token", "assistant_delta"}:
                stream_key = (_field(event, "request_id"), _field(event, "epoch"))
                if stream_key != self._answer_stream_key:
                    self.transcript.appendPlainText("Assistant: ")
                    self._answer_stream_key = stream_key
                self._append_inline(str(payload.get("text", payload.get("value", ""))))
            elif kind in {"final", "answer"}:
                text = payload.get("text", payload.get("answer", ""))
                if payload.get("streamed"):
                    self._append_inline("\n")
                else:
                    self.transcript.appendPlainText(f"Assistant: {text}\n")
            elif kind == "route":
                label = f"Route: {payload.get('model', 'unreported')}"
                if payload.get("experimental"):
                    label += " — EXPERIMENTAL; no qualified default or p95 yet"
                elif isinstance(payload.get("qualified_p95_s"), (int, float)):
                    label += f" — measured full-answer p95 {payload['qualified_p95_s']:.2f}s"
                self.route_label.setText(label)
                self.route_label.setToolTip(f"Route ID: {payload.get('route_id', 'unreported')}")
                self.placement_label.setText(_placement_text(payload))
                self.transcript.appendPlainText("[route] " + label.removeprefix("Route: "))
                if payload.get("selection_refusals"):
                    self.transcript.appendPlainText(
                        "[other routes] " + json.dumps(payload["selection_refusals"], ensure_ascii=False)
                    )
            elif kind == "model_metrics":
                self.transcript.appendPlainText("[model] " + _model_metric_summary(payload))
            elif kind == "approval_required":
                challenge = payload.get("challenge", payload)
                if hasattr(challenge, "to_dict"):
                    challenge = challenge.to_dict()
                if isinstance(challenge, Mapping):
                    challenge_id = challenge.get("challenge_id")
                    if isinstance(challenge_id, str) and challenge_id:
                        action = challenge.get("action")
                        action = action if isinstance(action, Mapping) else {}
                        operation = challenge.get("operation", action.get("operation"))
                        review = None
                        if operation == "browser_post":
                            try:
                                review = _browser_post_review(challenge)
                            except ValueError as exc:
                                detail = f"POST approval cannot be reviewed: {exc}"
                                self.transcript.appendPlainText(f"[approval error] {detail}")
                                self.status_label.setText(detail)
                                return
                            target = challenge["target"]
                            display = (
                                "HTTP POST — exact request approval required\n"
                                f"Target URL: {target['url']}\n"
                                f"Content-Type: {target['content_type']}\n"
                                f"Body: {target['body_size']} bytes; SHA-256 {target['body_sha256']}\n"
                                "Select Approve to inspect every payload byte."
                            )
                        elif operation == "screen_capture":
                            try:
                                review = _screen_capture_review(challenge)
                            except ValueError as exc:
                                detail = f"Screen capture approval cannot be reviewed: {exc}"
                                self.transcript.appendPlainText(f"[approval error] {detail}")
                                self.status_label.setText(detail)
                                return
                            display = review + "\nSelect Approve to confirm this one capture."
                        else:
                            display = (
                                f"{challenge.get('risk', 'action')}: {challenge.get('description', '')}\n"
                                f"Target: {json.dumps(challenge.get('target', {}), ensure_ascii=False)}\n"
                                f"Requested {operation or 'operation'}: "
                                f"{json.dumps(challenge.get('arguments', {}), ensure_ascii=False)}"
                            )
                        item = QListWidgetItem(display)
                        item.setData(Qt.ItemDataRole.UserRole, challenge_id)
                        item.setData(Qt.ItemDataRole.UserRole + 1, challenge.get("risk"))
                        item.setData(Qt.ItemDataRole.UserRole + 2, review)
                        item.setData(Qt.ItemDataRole.UserRole + 3, operation)
                        if operation == "screen_capture":
                            item.setData(Qt.ItemDataRole.UserRole + 4, dict(challenge["target"]))
                        self.approvals.addItem(item)
                        self.approvals.setCurrentItem(item)
                        self.approve_button.setEnabled(True)
                        self.reject_button.setEnabled(True)
                        self.status_label.setText("Waiting for your approval")
            elif kind == "refusal":
                self.transcript.appendPlainText(
                    "[refusal] " + str(payload.get("message", payload.get("reason", "route refused")))
                )
                if payload.get("reasons"):
                    self.transcript.appendPlainText(
                        "[admission reasons] " + json.dumps(payload["reasons"], ensure_ascii=False)
                    )
            elif kind == "error":
                self.transcript.appendPlainText("[error] " + str(payload.get("message", payload)))
            elif kind == "tool_result":
                self.transcript.appendPlainText("[tool] " + json.dumps(_display_payload(payload), ensure_ascii=False))
            else:
                self.transcript.appendPlainText(
                    f"[{kind}] " + json.dumps(_display_payload(payload), ensure_ascii=False)
                )
            if kind in {"final", "answer", "refusal", "error", "cancelled"} or bool(_field(event, "terminal", False)):
                self.approvals.clear()
                self.approve_button.setEnabled(True)
                self.reject_button.setEnabled(True)
                self._active = False
                self.send_button.setEnabled(True)
                self.read_url_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                self.clear_memory_button.setEnabled(True)
                self.status_label.setText({
                    "final": "Ready", "answer": "Ready", "cancelled": "Cancelled",
                    "refusal": "Request refused", "error": "Request failed",
                }.get(kind, kind))

        def _report_callback_error(self, message: str) -> None:
            self.transcript.appendPlainText(f"[controller error] {message}")
            self.approvals.clear()
            self.status_label.setText(message)
            self._active = False
            self.send_button.setEnabled(True)
            self.read_url_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.clear_memory_button.setEnabled(True)


def run_desktop(controller: DesktopController) -> int:
    """Launch the native window after the caller constructs an Omni controller."""
    if QApplication is None:
        raise RuntimeError("PySide6 is required for the native Windows Agent UI")
    app = QApplication.instance() or QApplication([])
    window = AgentWindow(controller)
    window.show()
    return int(app.exec())
