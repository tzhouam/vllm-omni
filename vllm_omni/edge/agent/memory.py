"""Encrypted, provenance-preserving local memory for the Omni edge agent.

The database never stores observed text, URLs, or request identifiers in clear
text.  Its search index consists of keyed hashes of normalized English words
and Chinese characters/bigrams.  Windows uses user-scoped DPAPI by default;
other platforms must supply an authenticated cipher explicitly.

This module owns persistence only.  The agent controller decides which stream
events are observations and calls :meth:`EncryptedMemoryStore.append_event`.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


_SCHEMA_VERSION = 3
_WORD_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
_CJK_RE = re.compile(r"^[\u3400-\u9fff]+$")
_BINARY_PAYLOAD_KEYS = frozenset({"base64", "$bytes_b64", "image_data_url"})


class EncryptionUnavailable(RuntimeError):
    """Persistent memory cannot be opened without authenticated encryption."""


class MemoryIntegrityError(RuntimeError):
    """Encrypted memory contents or indexed metadata failed integrity checks."""


class EventSequenceError(ValueError):
    """An event would reuse or rewind a request/epoch output sequence."""


class MemoryCipher(Protocol):
    name: str

    def encrypt(self, plaintext: bytes, associated_data: bytes) -> bytes: ...

    def decrypt(self, ciphertext: bytes, associated_data: bytes) -> bytes: ...


class AesGcmCipher:
    """Injectable authenticated cipher, e.g. with a key from an OS keyring.

    The caller is responsible for durable, secure key storage.  The Windows
    default uses DPAPI instead and does not require this optional dependency.
    """

    name = "aes-256-gcm"

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("AES-256-GCM requires a 32-byte key")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise EncryptionUnavailable(
                "cryptography is required for an injected AES-GCM cipher"
            ) from exc
        self._aes = AESGCM(key)

    def encrypt(self, plaintext: bytes, associated_data: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self._aes.encrypt(nonce, plaintext, associated_data)

    def decrypt(self, ciphertext: bytes, associated_data: bytes) -> bytes:
        if len(ciphertext) < 28:
            raise MemoryIntegrityError("encrypted record is truncated")
        try:
            return self._aes.decrypt(
                ciphertext[:12], ciphertext[12:], associated_data
            )
        except Exception as exc:
            raise MemoryIntegrityError("encrypted record failed authentication") from exc


class WindowsDpapiCipher:
    """Encrypt as the current Windows user, without installing a Python wheel."""

    name = "windows-dpapi-user"
    _UI_FORBIDDEN = 0x1

    def __init__(self) -> None:
        if os.name != "nt":
            raise EncryptionUnavailable("Windows DPAPI is available only on Windows")

    @staticmethod
    def _blob(data: bytes) -> tuple[Any, Any]:
        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
            ]

        backing = ctypes.create_string_buffer(data)
        blob = DataBlob(
            len(data), ctypes.cast(backing, ctypes.POINTER(ctypes.c_ubyte))
        )
        return blob, backing

    def _call(self, data: bytes, associated_data: bytes, *, protect: bool) -> bytes:
        # DPAPI authenticates its ciphertext.  Optional entropy binds each
        # record to its random event ID, preventing row substitution.
        if not associated_data:
            raise ValueError("DPAPI associated data must not be empty")
        from ctypes import wintypes

        in_blob, in_backing = self._blob(data)
        entropy_blob, entropy_backing = self._blob(associated_data)
        out_blob, _ = self._blob(b"")
        crypt32 = ctypes.windll.crypt32
        if protect:
            fn = crypt32.CryptProtectData
            fn.argtypes = [
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ]
            args = (
                ctypes.byref(in_blob),
                None,
                ctypes.byref(entropy_blob),
                None,
                None,
                self._UI_FORBIDDEN,
                ctypes.byref(out_blob),
            )
        else:
            fn = crypt32.CryptUnprotectData
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ]
            args = (
                ctypes.byref(in_blob),
                None,
                ctypes.byref(entropy_blob),
                None,
                None,
                self._UI_FORBIDDEN,
                ctypes.byref(out_blob),
            )
        fn.restype = wintypes.BOOL
        # Keep ctypes-owned input buffers alive through the native call.
        _ = (in_backing, entropy_backing)
        if not fn(*args):
            raise MemoryIntegrityError(
                f"Windows DPAPI {'encryption' if protect else 'decryption'} failed: "
                f"{ctypes.WinError()}"
            )
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(
                ctypes.cast(out_blob.pbData, ctypes.c_void_p)
            )

    def encrypt(self, plaintext: bytes, associated_data: bytes) -> bytes:
        return self._call(plaintext, associated_data, protect=True)

    def decrypt(self, ciphertext: bytes, associated_data: bytes) -> bytes:
        return self._call(ciphertext, associated_data, protect=False)


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    session_id: str
    request_id: str
    epoch: int
    sequence: int
    kind: str
    payload: Any
    source: Any
    observed_at: str
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryMatch:
    event: MemoryEvent
    score: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"memory payload is not JSON-compatible: {type(value).__name__}")


def _text_fragments(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        if set(value) == {"$bytes_b64"}:
            return
        for key, item in value.items():
            if key in _BINARY_PAYLOAD_KEYS:
                continue
            yield str(key)
            yield from _text_fragments(item)
    elif isinstance(value, list):
        for item in value:
            yield from _text_fragments(item)


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    found: set[str] = set()
    for token in _WORD_RE.findall(normalized):
        if _CJK_RE.fullmatch(token):
            found.update(token)
            found.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            found.add(token)
    return found


def _event_terms(event: MemoryEvent) -> set[str]:
    text = " ".join(
        fragment
        for item in (event.kind, event.source, event.payload)
        for fragment in _text_fragments(item)
    )
    return _terms(text)


class EncryptedMemoryStore:
    """Durable agent observations with encrypted contents and indexed recall.

    `session_id`, `request_id`, `epoch`, and `sequence` come from the Omni
    streaming contract.  Replaying the same event is idempotent; reusing a
    sequence for different content is an error.  A derived memory can list
    source event IDs; deleting any source also deletes all descendants and
    their search entries in the same transaction.
    """

    def __init__(self, path: str | Path, cipher: MemoryCipher | None = None) -> None:
        if cipher is None:
            cipher = WindowsDpapiCipher() if os.name == "nt" else None
        if cipher is None:
            raise EncryptionUnavailable(
                "Encrypted memory requires Windows DPAPI or an injected "
                "authenticated cipher backed by secure key storage"
            )
        self._cipher = cipher
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), timeout=30, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA secure_delete=ON")
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._initialize()
        except Exception:
            self._conn.close()
            raise

    def _initialize(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    name TEXT PRIMARY KEY, value BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_tag BLOB NOT NULL,
                    request_tag BLOB NOT NULL,
                    epoch INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    observed_us INTEGER NOT NULL,
                    ciphertext BLOB NOT NULL,
                    UNIQUE(request_tag, epoch, sequence)
                );
                CREATE INDEX IF NOT EXISTS events_session
                    ON events(session_tag, observed_us);
                CREATE TABLE IF NOT EXISTS term_index (
                    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
                    term_tag BLOB NOT NULL,
                    PRIMARY KEY(event_id, term_tag)
                );
                CREATE INDEX IF NOT EXISTS terms_lookup ON term_index(term_tag);
                CREATE TABLE IF NOT EXISTS event_kinds (
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
                    kind_tag BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_kinds_lookup
                    ON event_kinds(kind_tag, event_id);
                CREATE TABLE IF NOT EXISTS event_content (
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
                    content_tag BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_content_lookup
                    ON event_content(content_tag, event_id);
                CREATE TABLE IF NOT EXISTS derivations (
                    derived_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
                    PRIMARY KEY(derived_id, source_id)
                );
                CREATE INDEX IF NOT EXISTS derivations_source
                    ON derivations(source_id);
                """
            )
            version = self._conn.execute(
                "SELECT value FROM meta WHERE name='schema_version'"
            ).fetchone()
            if version is None:
                if self._conn.execute("SELECT 1 FROM events LIMIT 1").fetchone():
                    raise MemoryIntegrityError("unversioned memory database")
                self._conn.execute(
                    "INSERT INTO meta(name,value) VALUES('schema_version',?)",
                    (str(_SCHEMA_VERSION).encode(),),
                )
                self._conn.execute(
                    "INSERT INTO meta(name,value) VALUES('cipher_name',?)",
                    (self._cipher.name.encode(),),
                )
                self._conn.execute(
                    "INSERT INTO meta(name,value) VALUES('index_key',?)",
                    (self._cipher.encrypt(secrets.token_bytes(32), b"index-key-v1"),),
                )
            elif version[0] not in {b"1", b"2", str(_SCHEMA_VERSION).encode()}:
                raise MemoryIntegrityError("unsupported memory schema version")
            name = self._conn.execute(
                "SELECT value FROM meta WHERE name='cipher_name'"
            ).fetchone()
            if name is None or name[0] != self._cipher.name.encode():
                raise EncryptionUnavailable("memory database uses a different cipher")
            protected_key = self._conn.execute(
                "SELECT value FROM meta WHERE name='index_key'"
            ).fetchone()
            if protected_key is None:
                raise MemoryIntegrityError("memory index key is missing")
            self._index_key = self._cipher.decrypt(protected_key[0], b"index-key-v1")
            if len(self._index_key) != 32:
                raise MemoryIntegrityError("memory index key has invalid length")
            if version is not None and version[0] in {b"1", b"2"}:
                # Backfill keyed metadata from authenticated ciphertext in one
                # transaction. A failed upgrade keeps the old version and can
                # be retried without losing any observations.
                self._conn.execute("DELETE FROM event_kinds")
                self._conn.execute("DELETE FROM event_content")
                for row in self._conn.execute("SELECT * FROM events"):
                    event = self._decode(row)
                    self._conn.execute(
                        "INSERT INTO event_kinds(event_id,kind_tag) VALUES(?,?)",
                        (event.event_id, self._tag("kind", event.kind)),
                    )
                    self._conn.execute(
                        "INSERT INTO event_content(event_id,content_tag) VALUES(?,?)",
                        (event.event_id, self._content_tag(event)),
                    )
                self._conn.execute(
                    "UPDATE meta SET value=? WHERE name='schema_version'",
                    (str(_SCHEMA_VERSION).encode(),),
                )

    def _tag(self, domain: str, value: str) -> bytes:
        return hmac.new(
            self._index_key,
            domain.encode() + b"\0" + value.encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def _request_tag(self, session_id: str, request_id: str) -> bytes:
        return self._tag("request", session_id + "\0" + request_id)

    def _content_tag(self, event: MemoryEvent) -> bytes:
        # Request identity and time do not change the observed content. Keep
        # the full canonical payload: a question without an answer must never
        # collapse into an otherwise similar observation containing that fact.
        content = json.dumps(
            [event.kind, event.source, event.payload],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return self._tag("content", content)

    @staticmethod
    def _observed_us(value: str) -> int:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("observed_at requires a timezone")
        return int(parsed.timestamp() * 1_000_000)

    def _decode(self, row: sqlite3.Row) -> MemoryEvent:
        event_id = row["event_id"]
        try:
            data = json.loads(
                self._cipher.decrypt(row["ciphertext"], event_id.encode("ascii"))
            )
            data["derived_from"] = tuple(data.get("derived_from", ()))
            event = MemoryEvent(**data)
        except Exception as exc:
            raise MemoryIntegrityError("memory event failed to decrypt") from exc
        if (
            event.event_id != event_id
            or event.epoch != row["epoch"]
            or event.sequence != row["sequence"]
            or self._tag("session", event.session_id) != row["session_tag"]
            or self._request_tag(event.session_id, event.request_id)
            != row["request_tag"]
            or self._observed_us(event.observed_at) != row["observed_us"]
        ):
            raise MemoryIntegrityError("memory event metadata was modified")
        return event

    def append_event(
        self,
        *,
        session_id: str,
        request_id: str,
        epoch: int,
        sequence: int,
        kind: str,
        payload: Any,
        source: Any,
        observed_at: str | None = None,
        derived_from: Sequence[str] = (),
    ) -> MemoryEvent:
        if not session_id or not request_id or not kind:
            raise ValueError("session_id, request_id, and kind are required")
        if epoch < 0 or sequence < 0:
            raise ValueError("epoch and sequence must be nonnegative")
        if observed_at is None:
            observed_at = datetime.now(timezone.utc).isoformat()
        observed_us = self._observed_us(observed_at)
        sources = tuple(dict.fromkeys(derived_from))
        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            request_id=request_id,
            epoch=epoch,
            sequence=sequence,
            kind=kind,
            payload=_json_safe(payload),
            source=_json_safe(source),
            observed_at=observed_at,
            derived_from=sources,
        )
        session_tag = self._tag("session", session_id)
        request_tag = self._request_tag(session_id, request_id)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT * FROM events WHERE request_tag=? AND epoch=? AND sequence=?",
                    (request_tag, epoch, sequence),
                ).fetchone()
                if existing is not None:
                    prior = self._decode(existing)
                    if (
                        prior.kind != event.kind
                        or prior.payload != event.payload
                        or prior.source != event.source
                        or prior.derived_from != event.derived_from
                    ):
                        raise EventSequenceError("sequence already contains different content")
                    self._conn.commit()
                    return prior
                latest = self._conn.execute(
                    "SELECT MAX(sequence) FROM events WHERE request_tag=? AND epoch=?",
                    (request_tag, epoch),
                ).fetchone()[0]
                if latest is not None and sequence <= latest:
                    raise EventSequenceError("request/epoch sequence moved backwards")
                for source_id in sources:
                    if not self._conn.execute(
                        "SELECT 1 FROM events WHERE event_id=?", (source_id,)
                    ).fetchone():
                        raise ValueError(f"source event does not exist: {source_id}")
                serialized = json.dumps(
                    event.__dict__, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                encrypted = self._cipher.encrypt(serialized, event.event_id.encode("ascii"))
                self._conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?)",
                    (
                        event.event_id, session_tag, request_tag, epoch,
                        sequence, observed_us, encrypted,
                    ),
                )
                self._conn.executemany(
                    "INSERT INTO term_index(event_id,term_tag) VALUES(?,?)",
                    (
                        (event.event_id, self._tag("term", term))
                        for term in _event_terms(event)
                    ),
                )
                self._conn.execute(
                    "INSERT INTO event_kinds(event_id,kind_tag) VALUES(?,?)",
                    (event.event_id, self._tag("kind", event.kind)),
                )
                self._conn.execute(
                    "INSERT INTO event_content(event_id,content_tag) VALUES(?,?)",
                    (event.event_id, self._content_tag(event)),
                )
                self._conn.executemany(
                    "INSERT INTO derivations(derived_id,source_id) VALUES(?,?)",
                    ((event.event_id, source_id) for source_id in sources),
                )
                self._conn.commit()
                return event
            except Exception:
                self._conn.rollback()
                raise

    def get_event(self, event_id: str) -> MemoryEvent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            return self._decode(row) if row is not None else None

    def iter_events(
        self, *, session_id: str | None = None, request_id: str | None = None
    ) -> list[MemoryEvent]:
        if request_id is not None and session_id is None:
            raise ValueError("request_id filtering requires session_id")
        where = []
        args: list[Any] = []
        if session_id is not None:
            where.append("session_tag=?")
            args.append(self._tag("session", session_id))
        if request_id is not None:
            where.append("request_tag=?")
            args.append(self._request_tag(session_id, request_id))
        clause = " WHERE " + " AND ".join(where) if where else ""
        order = "epoch,sequence" if request_id is not None else "observed_us,event_id"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events" + clause + " ORDER BY " + order, args
            ).fetchall()
            return [self._decode(row) for row in rows]

    def search(
        self, query: str, *, session_id: str | None = None, limit: int = 10,
        kinds: frozenset[str] | None = None,
        exclude_event_ids: frozenset[str] = frozenset(),
    ) -> list[MemoryMatch]:
        """Return source-weighted matches without exposing cleartext index terms.

        Exclusions are applied inside the SQL candidate query, so the current
        request cannot spend one of the limited recall slots on itself.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query_terms = _terms(query)
        if not query_terms:
            return []
        if kinds is not None and not kinds:
            return []
        term_tags = [self._tag("term", term) for term in query_terms]
        placeholders = ",".join("?" for _ in term_tags)
        where = ""
        filter_args: list[Any] = []
        if session_id is not None:
            where += " AND e.session_tag=?"
            filter_args.append(self._tag("session", session_id))
        excluded = tuple(exclude_event_ids)
        if excluded:
            where += " AND e.event_id NOT IN (" + ",".join("?" for _ in excluded) + ")"
            filter_args.extend(excluded)
        if kinds is not None:
            kind_tags = [self._tag("kind", kind) for kind in sorted(kinds)]
            where += " AND k.kind_tag IN (" + ",".join("?" for _ in kind_tags) + ")"
            filter_args.extend(kind_tags)
        # Rank and filter by keyed kind tags before decrypting candidates.
        # Identical observations across requests take one candidate slot:
        # repeated recall questions must not crowd out an older source fact.
        # This exact-content grouping cannot solve paraphrase saturation.
        candidate_limit = max(100, limit * 4)
        sql = (
            "WITH term_frequency AS ("
            " SELECT term_tag, COUNT(*) AS documents FROM term_index"
            f" WHERE term_tag IN ({placeholders}) GROUP BY term_tag"
            "), ranked AS (SELECT e.*, k.kind_tag, c.content_tag, "
            "(CASE WHEN k.kind_tag=? THEN 0.25 ELSE 1.0 END) * "
            "SUM(1.0 / f.documents / f.documents / f.documents / f.documents) AS weighted_hits "
            "FROM events e JOIN term_index i ON i.event_id=e.event_id "
            "JOIN term_frequency f ON f.term_tag=i.term_tag "
            "JOIN event_kinds k ON k.event_id=e.event_id "
            "JOIN event_content c ON c.event_id=e.event_id "
            f"WHERE 1=1{where} "
            "GROUP BY e.event_id), deduped AS ("
            "SELECT ranked.*, ROW_NUMBER() OVER ("
            "PARTITION BY content_tag "
            "ORDER BY weighted_hits DESC,observed_us DESC,event_id DESC"
            ") AS content_rank FROM ranked) "
            "SELECT * FROM deduped WHERE content_rank=1 "
            "ORDER BY weighted_hits DESC,observed_us DESC,event_id DESC LIMIT ?"
        )
        args: list[Any] = [*term_tags, self._tag("kind", "final"),
                           *filter_args, candidate_limit]
        matches: list[MemoryMatch] = []
        with self._lock:
            frequencies = {
                row[0]: row[1] for row in self._conn.execute(
                    "SELECT term_tag,COUNT(*) FROM term_index WHERE term_tag IN ("
                    + placeholders + ") GROUP BY term_tag", term_tags,
                )
            }
            weights = {
                term: (1.0 / frequencies[self._tag("term", term)]) ** 4
                for term in query_terms if self._tag("term", term) in frequencies
            }
            total_weight = sum(weights.values())
            for row in self._conn.execute(sql, args):
                event = self._decode(row)
                if (row["kind_tag"] != self._tag("kind", event.kind)
                        or row["content_tag"] != self._content_tag(event)):
                    raise MemoryIntegrityError("encrypted memory search index disagrees with event")
                if kinds is not None and event.kind not in kinds:
                    raise MemoryIntegrityError("encrypted memory kind index disagrees with event")
                intersection = query_terms & _event_terms(event)
                if intersection:
                    # A generated answer may echo a prior observation. Keep
                    # it searchable, but prefer the direct user/tool source
                    # when both describe the same fact.
                    origin_weight = 0.25 if event.kind == "final" else 1.0
                    matches.append(
                        MemoryMatch(event=event, score=(
                            sum(weights.get(term, 0.0) for term in intersection)
                            / total_weight
                        ) * origin_weight)
                    )
        # The encrypted index can contain ordinary conversation questions and
        # derivative Agent outputs as well as source observations. Rare query
        # terms carry more evidence than generic bilingual instructions; a
        # source with an exact distinctive name must not lose to recent echoes
        # of "recall from memory". Fourth-power frequency suppression was
        # chosen for the local event index so a few rare entity terms
        # outrank dozens of repeated prompt-instruction terms. SQL and this
        # plaintext recheck use the same weight, without storing cleartext
        # terms. Generated final answers are then demoted by origin weight.
        matches.sort(key=lambda match: (
            match.score,
            self._observed_us(match.event.observed_at),
            match.event.event_id,
        ), reverse=True)
        return matches[:limit]

    def _delete_ids(self, initial_ids: Sequence[str]) -> int:
        if not initial_ids:
            return 0
        # Conservative provenance rule: a summary depending on a deleted
        # observation is also deleted, even if it cited other observations.
        placeholders = ",".join("?" for _ in initial_ids)
        rows = self._conn.execute(
            "WITH RECURSIVE doomed(id) AS ("
            f" SELECT event_id FROM events WHERE event_id IN ({placeholders})"
            " UNION SELECT d.derived_id FROM derivations d "
            " JOIN doomed ON d.source_id=doomed.id"
            ") SELECT id FROM doomed",
            list(initial_ids),
        ).fetchall()
        ids = [row[0] for row in rows]
        self._conn.executemany(
            "DELETE FROM events WHERE event_id=?", ((event_id,) for event_id in ids)
        )
        return len(ids)

    def delete_event(self, event_id: str) -> int:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                count = self._delete_ids([event_id])
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def delete_request(self, session_id: str, request_id: str) -> int:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                ids = [
                    row[0] for row in self._conn.execute(
                        "SELECT event_id FROM events WHERE request_tag=?",
                        (self._request_tag(session_id, request_id),),
                    )
                ]
                count = self._delete_ids(ids)
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                ids = [
                    row[0] for row in self._conn.execute(
                        "SELECT event_id FROM events WHERE session_tag=?",
                        (self._tag("session", session_id),),
                    )
                ]
                count = self._delete_ids(ids)
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def delete_all(self) -> int:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                count = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                self._conn.execute("DELETE FROM events")
                self._conn.commit()
                self._conn.execute("VACUUM")
                return count
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> EncryptedMemoryStore:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
