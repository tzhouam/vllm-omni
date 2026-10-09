# SPDX-License-Identifier: Apache-2.0
"""Opt-in, app-owned declaration for one exactly observed CDP helper.

The descriptor binds named installation files, not loaded image contents or
OS ancestry. CDP data cannot construct or broaden it. It grants observation
and retained-handle retirement checks only, never process termination.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IMAGE_MAX_BYTES = 256 * 1024 * 1024  # Finite artifact verification bound, not a RAM allowance.


def _path_key(value: Any) -> str:
    if (type(value) is not str or not 1 <= len(value) <= 4096
            or not value.isprintable() or any(char in value for char in '*?<>|"')):
        raise ValueError("an exact bounded local Windows image path is required")
    drive, tail = ntpath.splitdrive(value)
    if (len(drive) != 2 or drive[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" or drive[1] != ":"
            or not tail.startswith(("\\", "/")) or ":" in tail
            or any(part in {".", ".."} for part in tail.replace("/", "\\").split("\\"))):
        raise ValueError("an absolute drive path without aliases is required")
    return ntpath.normcase(ntpath.normpath(value))


def _verify_named_image(path: str, expected_size: int, expected_sha256: str) -> None:
    """Read one declared file, with a finite bound and stable open-file stat.

    This is installation provenance at verification time. It cannot attest the
    bytes mapped into a process, prevent a later file change, or establish an
    exclusive relationship between a service and this browser.
    """
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != expected_size:
            raise ValueError("declared CDP image file size differs")
        digest = hashlib.sha256()
        remaining = expected_size + 1
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("declared CDP image file changed during verification")
        if remaining != 1 or digest.hexdigest() != expected_sha256:
            raise ValueError("declared CDP image file SHA256 differs")


@dataclass(frozen=True)
class CdpObservedHelperCapability:
    """One app-owned, exact installation/type declaration; absent by default.

    Construct only from reviewed application configuration. Never construct
    from page text, tool results or CDP ProcessInfo. A previous diagnostic
    receipt is rationale for review, not authority to release a current lease.
    Every currently bound helper must independently retire on its retained
    process object before the existing companion release contract can pass.
    """

    browser_image_path: str
    browser_size_bytes: int
    browser_sha256: str
    helper_image_path: str
    helper_size_bytes: int
    helper_sha256: str
    cdp_process_type: str
    declaration_sha256: str | None = None
    capability_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        browser, helper = _path_key(self.browser_image_path), _path_key(self.helper_image_path)
        root = ntpath.dirname(browser)
        if (ntpath.basename(browser) != "msedge.exe" or ntpath.basename(helper) == "msedge.exe"
                or ntpath.commonpath((root, helper)) != root):
            raise ValueError("helper and browser must name the same declared Edge installation")
        for size, digest in ((self.browser_size_bytes, self.browser_sha256),
                             (self.helper_size_bytes, self.helper_sha256)):
            if type(size) is not int or not 0 < size <= _IMAGE_MAX_BYTES:
                raise ValueError("an explicit bounded image file size is required")
            if (type(digest) is not str or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("an exact lowercase image SHA256 is required")
        if self.declaration_sha256 is not None and (
                type(self.declaration_sha256) is not str or len(self.declaration_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.declaration_sha256)):
            raise ValueError("an exact declaration SHA256 is required")
        label = self.cdp_process_type
        if (type(label) is not str or not 1 <= len(label) <= 128
                or not label.isprintable() or not label.strip()
                or len(label.encode("utf-8")) > 128
                or label in {"browser", "renderer", "GPU", "utility", "other"}):
            raise ValueError("an exact bounded helper CDP type is required")
        object.__setattr__(self, "browser_image_path", browser)
        object.__setattr__(self, "helper_image_path", helper)
        raw = json.dumps(self._declaration(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        object.__setattr__(self, "capability_sha256", hashlib.sha256(raw).hexdigest())

    def _declaration(self) -> dict[str, Any]:
        return {"schema": "omni-cdp-observed-helper-capability-v1",
                "browser_image_path": self.browser_image_path,
                "browser_size_bytes": self.browser_size_bytes, "browser_sha256": self.browser_sha256,
                "helper_image_path": self.helper_image_path,
                "helper_size_bytes": self.helper_size_bytes, "helper_sha256": self.helper_sha256,
                "cdp_process_type": self.cdp_process_type,
                **({"declaration_sha256": self.declaration_sha256}
                   if self.declaration_sha256 is not None else {})}

    def metadata(self) -> dict[str, Any]:
        return {**self._declaration(), "capability_sha256": self.capability_sha256,
                "scope": "declared_CDP_observed_helper_same_retained_process_object",
                "named_file_hash_is_not_loaded_image_attestation": True,
                "ancestry_verified": False, "exclusive_ownership_verified": False,
                "termination_authority": False, "retirement_required": True}

    def verify_named_artifacts(self) -> None:
        _verify_named_image(self.browser_image_path, self.browser_size_bytes, self.browser_sha256)
        _verify_named_image(self.helper_image_path, self.helper_size_bytes, self.helper_sha256)

    def matches(self, *, browser_image_path: str, helper_image_path: str,
                cdp_process_type: str) -> bool:
        return (cdp_process_type == self.cdp_process_type
                and _path_key(browser_image_path) == self.browser_image_path
                and _path_key(helper_image_path) == self.helper_image_path)


def resolve_cdp_observed_helper_capability(path: Path, *, expected_sha256: str) -> CdpObservedHelperCapability:
    """Resolve only explicit app configuration, never a tool/CDP payload.

    This bounded local descriptor grants no lease, chooses no RAM/GPU allowance
    and does not consume a previous diagnostic receipt as current authority.
    Named image files are verified now and again at actual registry adoption.
    """
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("an explicit lowercase helper declaration SHA256 is required")
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024 + 1)
    if len(raw) > 16 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("helper declaration exceeds its bound or hash differs")
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {}
        for key, value in pairs:
            if key in body:
                raise ValueError("duplicate helper declaration JSON field")
            body[key] = value
        return body
    body = json.loads(raw, object_pairs_hook=unique_fields)
    fields = {"browser_image_path", "browser_size_bytes", "browser_sha256",
              "helper_image_path", "helper_size_bytes", "helper_sha256", "cdp_process_type"}
    if (type(body) is not dict or set(body) != fields | {"schema"}
            or body["schema"] != "omni-cdp-observed-helper-capability-v1"):
        raise ValueError("an exact app-owned helper declaration schema is required")
    capability = CdpObservedHelperCapability(
        **{name: body[name] for name in fields}, declaration_sha256=expected_sha256,
    )
    capability.verify_named_artifacts()
    return capability
