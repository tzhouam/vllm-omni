# SPDX-License-Identifier: Apache-2.0
"""The qualification timing contract for one local request at a time."""

PROTOCOL = "single_request_batch1_v1"
LENGTH_BANDS = ("short", "medium", "long")
MIN_MEASURED_PER_BAND = 20
MIN_SUSTAINED_SECONDS = 1800


def validate_settings(batch_size: int, concurrency: int) -> None:
    """Reject a performance profile that could run more than one request at once."""
    if batch_size != 1 or concurrency != 1:
        raise ValueError(
            "Qualification profiling requires batch size 1 and concurrency 1; "
            f"got batch_size={batch_size}, concurrency={concurrency}"
        )


def metadata() -> dict:
    return {
        "name": PROTOCOL,
        "request_batch_size": 1,
        "max_active_requests": 1,
        "length_bands": list(LENGTH_BANDS),
        "minimum_measured_per_band": MIN_MEASURED_PER_BAND,
        "minimum_sustained_seconds": MIN_SUSTAINED_SECONDS,
        "sustained_request_mode": "sequential",
    }
