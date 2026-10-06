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
            assert conn.execute("SELECT COUNT(*) FROM event_kinds").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM event_content").fetchone()[0] == 0
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


def test_rare_bilingual_entity_outweighs_repeated_recall_boilerplate(tmp_path):
    with _store(tmp_path) as store:
        target = store.append_event(
            session_id="earlier", request_id="source", epoch=0, sequence=0,
            kind="user_observation",
            payload={"text": "Amber compass (琥珀罗盘) identifier is IDA19F"},
            source="user",
        )
        distractor = store.append_event(
            session_id="earlier", request_id="distractor", epoch=0, sequence=0,
            kind="user_observation",
            payload={"text": "Amber notebook (琥珀笔记) identifier is IDB28E"},
            source="user",
        )
        for index in range(6):
            store.append_event(
                session_id="recent", request_id=f"question-{index}",
                epoch=0, sequence=0, kind="user_observation",
                payload={"text": (
                    "请从本地记忆中回忆别的条目标识符。只依据完全匹配的观察，"
                    "不要使用相似条目。如果该观察不存在，只回答 UNKNOWN。"
                )}, source="user",
            )
        query = (
            "请从本地记忆中回忆琥珀罗盘（Amber compass）的标识符。"
            "只依据完全匹配的观察，不要使用相似条目。"
        )
        current = store.append_event(
            session_id="recent", request_id="current", epoch=0, sequence=0,
            kind="user_observation", payload={"text": query}, source="user",
        )
        ranked = store.search(
            query, kinds=frozenset({"user_observation", "final"}),
            exclude_event_ids=frozenset({current.event_id}), limit=5,
        )
        ids = [match.event.event_id for match in ranked]
        assert current.event_id not in ids
        assert target.event_id in ids
        assert distractor.event_id in ids


def test_exact_recall_question_echoes_do_not_hide_older_distinct_facts(tmp_path):
    with _store(tmp_path) as store:
        sources = [
            store.append_event(
                session_id="old", request_id=f"fact-{index}", epoch=0, sequence=0,
                kind="user_observation",
                payload={"text": f"Amber compass identifier is {identifier}"},
                source="user",
            )
            for index, identifier in enumerate(("IDA19F", "IDB28E"))
        ]
        question = "What is Amber compass identifier?"
        for index in range(160):
            store.append_event(
                session_id="new", request_id=f"question-{index}", epoch=0,
                sequence=0, kind="user_observation",
                payload={"text": question}, source="user",
            )
        ranked = store.search(
            question, kinds=frozenset({"user_observation"}), limit=5,
        )
        ids = {match.event.event_id for match in ranked}
        assert all(source.event_id in ids for source in sources)
        assert len([match for match in ranked if match.event.session_id == "new"]) == 1


def test_direct_observation_ranks_ahead_of_derivative_final(tmp_path):
    with _store(tmp_path) as store:
        original = store.append_event(
            session_id="s1", request_id="source", epoch=0, sequence=0,
            kind="user_observation", payload={"text": "Orchid key IDA19F"}, source="user",
        )
        echo = store.append_event(
            session_id="s2", request_id="answer", epoch=0, sequence=0,
            kind="final", payload={"answer": "Orchid key IDA19F"}, source="agent",
            derived_from=(original.event_id,),
        )
        ranked = store.search("Orchid key IDA19F", kinds=frozenset({
            "user_observation", "final",
        }))
        assert [match.event.event_id for match in ranked] == [
            original.event_id, echo.event_id,
        ]
        assert ranked[0].score > ranked[1].score


def test_v1_encrypted_memory_migrates_to_keyed_search_indexes(tmp_path):
    database = tmp_path / "agent.sqlite3"
    with _store(tmp_path) as store:
        source = store.append_event(
            session_id="s1", request_id="source", epoch=0, sequence=0,
            kind="user_observation", payload={"text": "Amber compass secret"},
            source="user",
        )
        store.append_event(
            session_id="s1", request_id="control", epoch=0, sequence=0,
            kind="approval_required", payload={"text": "Amber approval"},
            source="agent",
        )
    # A v1 database has the same encrypted event and term tables, without
    # keyed kind or content indexes. Migration backfills from local ciphertext.
    with sqlite3.connect(str(database)) as conn:
        conn.execute("DROP TABLE event_kinds")
        conn.execute("DROP TABLE event_content")
        conn.execute("UPDATE meta SET value=? WHERE name='schema_version'", (b"1",))
    with _store(tmp_path) as migrated:
        matches = migrated.search("Amber", kinds=frozenset({"user_observation"}))
        assert [match.event.event_id for match in matches] == [source.event_id]
    with sqlite3.connect(str(database)) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE name='schema_version'",
        ).fetchone()[0] == b"3"
        assert conn.execute("SELECT COUNT(*) FROM event_kinds").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM event_content").fetchone()[0] == 2
    assert b"user_observation" not in database.read_bytes()


def test_v2_encrypted_memory_migrates_content_index_without_losing_events(tmp_path):
    database = tmp_path / "agent.sqlite3"
    with _store(tmp_path) as store:
        original = _append(store, payload={"text": "orchid key IDA19F"})
    with sqlite3.connect(str(database)) as conn:
        conn.execute("DROP TABLE event_content")
        conn.execute("UPDATE meta SET value=? WHERE name='schema_version'", (b"2",))
    with _store(tmp_path) as migrated:
        matches = migrated.search("orchid", kinds=frozenset({"page_observation"}))
        assert [match.event.event_id for match in matches] == [original.event_id]
    with sqlite3.connect(str(database)) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE name='schema_version'",
        ).fetchone()[0] == b"3"
        assert conn.execute("SELECT COUNT(*) FROM event_content").fetchone()[0] == 1


@pytest.mark.skipif(os.name != "nt", reason="DPAPI requires native Windows")
def test_v1_dpapi_memory_migrates_without_losing_observations(tmp_path):
    database = tmp_path / "dpapi-agent.sqlite3"
    with EncryptedMemoryStore(database) as store:
        source = store.append_event(
            session_id="s1", request_id="source", epoch=0, sequence=0,
            kind="user_observation", payload={"text": "苍鹭包裹 identifier"},
            source="user",
        )
    with sqlite3.connect(str(database)) as conn:
        conn.execute("DROP TABLE event_kinds")
        conn.execute("DROP TABLE event_content")
        conn.execute("UPDATE meta SET value=? WHERE name='schema_version'", (b"1",))
    with EncryptedMemoryStore(database) as migrated:
        matches = migrated.search("苍鹭包裹", kinds=frozenset({"user_observation"}))
        assert [match.event.event_id for match in matches] == [source.event_id]


def test_kind_filter_bounds_decryptions_with_many_matching_control_events(tmp_path):
    with _store(tmp_path) as store:
        source = store.append_event(
            session_id="s1", request_id="source", epoch=0, sequence=0,
            kind="user_observation", payload={"text": "orchid identity"},
            source="user",
        )
        for index in range(160):
            store.append_event(
                session_id="s1", request_id=f"control-{index}", epoch=0,
                sequence=0, kind="approval_required",
                payload={"text": "orchid control"}, source="agent",
            )
        original_decode = store._decode
        decoded = []

        def counted(row):
            decoded.append(row["event_id"])
            return original_decode(row)

        store._decode = counted
        matches = store.search("orchid", kinds=frozenset({"user_observation"}),
                               limit=5)
        assert [match.event.event_id for match in matches] == [source.event_id]
        assert decoded == [source.event_id]


@pytest.mark.skipif(os.name == "nt", reason="Windows has DPAPI by default")
def test_default_encryption_fails_closed_outside_windows(tmp_path):
    with pytest.raises(EncryptionUnavailable):
        EncryptedMemoryStore(tmp_path / "memory.sqlite3")
    assert not (tmp_path / "memory.sqlite3").exists()
