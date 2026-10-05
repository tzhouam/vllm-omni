# SPDX-License-Identifier: Apache-2.0
"""Benchmark-facing alias for the wheel-packaged native Agent verifier.

The desktop application imports vllm_omni.edge.agent.qualification directly;
benchmark data and fixture servers are never included in the wheel.
"""

from vllm_omni.edge.agent.qualification import (
    GATE_SCHEMA,
    PROMOTION_SCHEMA,
    REQUIRED_GATES,
    EvidenceAudit,
    audit_summary,
    bundle_signing_bytes,
    load_reviewed_qualification,
)

__all__ = [
    "GATE_SCHEMA", "PROMOTION_SCHEMA", "REQUIRED_GATES", "EvidenceAudit",
    "audit_summary", "bundle_signing_bytes", "load_reviewed_qualification",
]
