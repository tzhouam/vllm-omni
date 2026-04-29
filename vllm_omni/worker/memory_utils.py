import math

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot, format_gib

logger = init_logger(__name__)


def request_memory_tolerant(
    init_snapshot: MemorySnapshot,
    cache_config: CacheConfig,
) -> int:
    """Compute the memory budget for a stage, tolerating shared-GPU scenarios.

    In multi-stage Omni pipelines, multiple stages can share the same
    physical GPU.  The upstream ``request_memory`` raises ``ValueError``
    when ``free_memory < gpu_memory_utilization * total_memory``, which
    always fails for later stages on a shared device.

    This helper caps the budget to the actually available free memory
    instead of raising, letting ``determine_available_memory`` in the
    base worker perform per-process accounting later.
    """
    requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)

    if init_snapshot.free_memory < requested_memory:
        logger.warning(
            "Free memory on %s (%s/%s GiB) is less than desired "
            "gpu_memory_utilization (%s, %s GiB). Capping to free "
            "memory — this is expected when multiple Omni stages "
            "share a GPU.",
            init_snapshot.device_,
            format_gib(init_snapshot.free_memory),
            format_gib(init_snapshot.total_memory),
            cache_config.gpu_memory_utilization,
            format_gib(requested_memory),
        )
        requested_memory = int(init_snapshot.free_memory)

    return requested_memory
