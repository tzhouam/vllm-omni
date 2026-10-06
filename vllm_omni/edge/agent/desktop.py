"""Optional PySide6 desktop shell for the local Omni edge agent.

The controller owns inference, memory and the tool policy.  This window only
subscribes to its ordered events and presents UI-only approval callbacks.
Importing this module on Linux or without PySide6 is safe; constructing the
window then reports the missing optional dependency plainly.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Mapping, Protocol

try:
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
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
            else _display_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_display_payload(item) for item in value]
    return value


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
            self.prompt.clear()
            self.transcript.appendPlainText(f"\nYou: {prompt}\n")
            self._active = True
            self.send_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.clear_memory_button.setEnabled(False)
            self.status_label.setText("Running one local request")
            self._invoke(self.controller.submit, prompt)

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
            if kind == "ValueError":
                self._remove_approval(challenge_id)
            self.approve_button.setEnabled(True)
            self.reject_button.setEnabled(True)
            detail = f"Approval failed ({kind}): {message}"
            self.transcript.appendPlainText(f"[approval error] {detail}")
            self.status_label.setText(detail)

        def _approve_selected(self) -> None:
            item = self.approvals.currentItem()
            if item is not None and item.data(Qt.ItemDataRole.UserRole + 1) == "high_impact_external_action":
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
            if kind in {"text_delta", "token", "assistant_delta"}:
                self._append_inline(str(payload.get("text", payload.get("value", ""))))
            elif kind in {"final", "answer"}:
                text = payload.get("text", payload.get("answer", ""))
                self._append_inline("\n" if payload.get("streamed") else f"{text}\n")
            elif kind == "route":
                label = (f"Route: {payload.get('model', 'unreported')} "
                         f"({payload.get('route_id', 'unreported')})")
                if payload.get("experimental"):
                    label += " — EXPERIMENTAL; no qualified default or p95 yet"
                elif isinstance(payload.get("qualified_p95_s"), (int, float)):
                    label += f" — measured full-answer p95 {payload['qualified_p95_s']:.2f}s"
                self.route_label.setText(label)
                placement = "Placement: " + json.dumps(
                    payload.get("actual_placement", "unreported"), ensure_ascii=False
                )
                if payload.get("placement_evidence_level") == "override_selection_only":
                    placement += " — expert override selected; final storage/compute unverified"
                self.placement_label.setText(placement)
                self.transcript.appendPlainText("[route] " + json.dumps(dict(payload), ensure_ascii=False))
                if payload.get("selection_refusals"):
                    self.transcript.appendPlainText(
                        "[other routes] " + json.dumps(payload["selection_refusals"], ensure_ascii=False)
                    )
            elif kind == "approval_required":
                challenge = payload.get("challenge", payload)
                if hasattr(challenge, "to_dict"):
                    challenge = challenge.to_dict()
                if isinstance(challenge, Mapping):
                    challenge_id = challenge.get("challenge_id")
                    if isinstance(challenge_id, str) and challenge_id:
                        display = (
                            f"{challenge.get('risk', 'action')}: {challenge.get('description', '')}\n"
                            f"Target: {json.dumps(challenge.get('target', {}), ensure_ascii=False)}\n"
                            f"Requested {challenge.get('operation', 'operation')}: "
                            f"{json.dumps(challenge.get('arguments', {}), ensure_ascii=False)}"
                        )
                        item = QListWidgetItem(display)
                        item.setData(Qt.ItemDataRole.UserRole, challenge_id)
                        item.setData(Qt.ItemDataRole.UserRole + 1, challenge.get("risk"))
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
                self.cancel_button.setEnabled(False)
                self.clear_memory_button.setEnabled(True)
                self.status_label.setText(kind)

        def _report_callback_error(self, message: str) -> None:
            self.transcript.appendPlainText(f"[controller error] {message}")
            self.approvals.clear()
            self.status_label.setText(message)
            self._active = False
            self.send_button.setEnabled(True)
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
