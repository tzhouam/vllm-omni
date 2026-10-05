"""Focused persistence, privacy, isolation, and deletion checks for memory."""

from __future__ import annotations

import os
import sqlite3

import pytest

from vllm_omni.edge.agent.memory import (
    AesGcmCipher,
    EncryptedMemoryStore,
    EncryptionUnavailable,
    EventSequenceError,
    MemoryIntegrityError,
)


def _store(tmp_path):
    return EncryptedMemoryStore(tmp_path / "agent.sqlite3", AesGcmCipher(b"k" * 32))


def _append(store, *, request="r1", sequence=0, payload=None, source=None, **extra):
    return store.append_event(
        session_id="s1",
        request_id=request,
        epoch=0,
        sequence=sequence,
        kind="page_observation",
        payload={"text": "打开浏览器设置 Browser settings"}
        if payload is None else payload,
        source={"url": "https://example.test/private"}
        if source is None else source,
        **extra,
    )


def test_encrypted_persistence_and_bilingual_search(tmp_path):
    database = tmp_path / "agent.sqlite3"
    with _store(tmp_path) as store:
        event = _append(store)
        assert store.search("浏览器设置")[0].event.event_id == event.event_id
        assert store.search("BROWSER settings")[0].event.event_id == event.event_id
        assert store.search("private")[0].event.event_id == event.event_id
        assert store.search("浏览器", session_id="other") == []
    raw = database.read_bytes()
    for sensitive in (b"Browser", "浏览器".encode(), b"private", b"s1", b"r1"):
        assert sensitive not in raw
    with _store(tmp_path) as reopened:
        assert reopened.get_event(event.event_id).payload["text"].endswith(
            "Browser settings"
        )


def test_sequence_idempotence_and_request_isolation(tmp_path):
    with _store(tmp_path) as store:
        first = _append(store)
        repeated = _append(store)
        assert repeated.event_id == first.event_id
        with pytest.raises(EventSequenceError):
            _append(store, payload={"text": "different"})
        _append(store, request="r2")
        _append(store, sequence=1)
        with pytest.raises(EventSequenceError):
            _append(store, sequence=0, payload={"text": "older"})
        assert len(store.iter_events(session_id="s1", request_id="r1")) == 2
        with pytest.raises(ValueError):
            store.iter_events(request_id="r1")


def test_derived_memory_and_index_deleted_with_source(tmp_path):
    with _store(tmp_path) as store:
        source = _append(store, payload={"text": "原始屏幕观察"})
        derived = store.append_event(
            session_id="s2",
            request_id="summary",
            epoch=0,
            sequence=0,
            kind="derived_memory",
            payload={"text": "用户喜欢深色模式"},
            source={"kind": "agent_summary"},
            derived_from=(source.event_id,),
        )
        descendant = store.append_event(
            session_id="s3",
            request_id="summary2",
            epoch=0,
            sequence=0,
            kind="derived_memory",
            payload={"text": "偏好深色模式的下游摘要"},
            source={"kind": "agent_summary"},
            derived_from=(derived.event_id,),
        )
        assert derived.event_id in {
            match.event.event_id for match in store.search("深色模式")
        }
        assert store.delete_request("s1", "r1") == 3
        assert store.get_event(source.event_id) is None
        assert store.get_event(derived.event_id) is None
        assert store.get_event(descendant.event_id) is None
        assert store.search("深色模式") == []
        with sqlite3.connect(str(tmp_path / "agent.sqlite3")) as conn:
            assert conn.execute("SELECT COUNT(*) FROM term_index").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM derivations").fetchone()[0] == 0


def test_bytes_are_encrypted_but_not_text_indexed(tmp_path):
    with _store(tmp_path) as store:
        event = _append(
            store,
            payload={"screenshot": b"private image pixels secretbase64token"},
            source={"kind": "screen"},
        )
        assert event.payload["screenshot"]["$bytes_b64"]
        assert store.search("secretbase64token") == []
    assert b"private image pixels" not in (tmp_path / "agent.sqlite3").read_bytes()


def test_base64_image_text_is_not_indexed_and_kind_filter_skips_control_events(tmp_path):
    with _store(tmp_path) as store:
        _append(store, sequence=0, payload={"base64": "secretbase64token",
                                            "text": "orchid from image"})
        store.append_event(
            session_id="s1", request_id="r1", epoch=0, sequence=1,
            kind="approval_required",
            payload={"challenge_id": "private-token", "text": "orchid approval"},
            source="agent",
        )
        assert store.search("secretbase64token") == []
        matches = store.search("orchid", kinds=frozenset({"page_observation"}))
        assert len(matches) == 1
        assert matches[0].event.kind == "page_observation"


def test_wrong_key_and_tampering_fail_closed(tmp_path):
    with _store(tmp_path) as store:
        event = _append(store)
    with pytest.raises(MemoryIntegrityError):
        EncryptedMemoryStore(tmp_path / "agent.sqlite3", AesGcmCipher(b"x" * 32))
    with sqlite3.connect(str(tmp_path / "agent.sqlite3")) as conn:
        conn.execute(
            "UPDATE events SET sequence=sequence+1 WHERE event_id=?", (event.event_id,)
        )
    with _store(tmp_path) as store:
        with pytest.raises(MemoryIntegrityError):
            store.get_event(event.event_id)


def test_explicit_deletion_and_no_implicit_retention(tmp_path):
    with _store(tmp_path) as store:
        one = _append(store)
        two = _append(store, request="r2")
        assert store.delete_event(one.event_id) == 1
        assert store.search("settings")[0].event.event_id == two.event_id
        assert store.delete_session("s1") == 1
        assert store.iter_events() == []
        _append(store, request="r3")
        assert store.delete_all() == 1
        assert store.search("浏览器") == []


@pytest.mark.skipif(os.name == "nt", reason="Windows has DPAPI by default")
def test_default_encryption_fails_closed_outside_windows(tmp_path):
    with pytest.raises(EncryptionUnavailable):
        EncryptedMemoryStore(tmp_path / "memory.sqlite3")
    assert not (tmp_path / "memory.sqlite3").exists()
