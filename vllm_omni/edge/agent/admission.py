# SPDX-License-Identifier: Apache-2.0
"""Compatibility names for the engine-owned local route manager.

The Agent holds an engine lease; it does not own a second memory ledger.
"""

from vllm_omni.engine.local_plan import (
    LocalPlanManager as HostMemoryCoordinator,
)
from vllm_omni.engine.local_plan import (
    ManagedLocalBackend as ManagedRouteBackend,
)

__all__ = ["HostMemoryCoordinator", "ManagedRouteBackend"]
