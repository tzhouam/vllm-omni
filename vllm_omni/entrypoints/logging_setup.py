import logging
import os
from logging.handlers import WatchedFileHandler
from typing import Optional


class StageContextFilter(logging.Filter):
    """Injects stage_id into LogRecord as 'stage' attribute."""

    def __init__(self, stage_id: int):
        super().__init__()
        self.stage_id = stage_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            setattr(record, "stage", self.stage_id)
        return True


class StageDefaultFilter(logging.Filter):
    """Sets default stage value when none was provided."""

    def __init__(self, default_stage: str = "-"):
        super().__init__()
        self.default_stage = default_stage

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            setattr(record, "stage", self.default_stage)
        return True


def _build_stage_file_handler(file_path: str, stage_id: int) -> logging.Handler:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    handler = WatchedFileHandler(file_path)
    formatter = logging.Formatter(
        "%(asctime)s [PID:%(process)d] [%(levelname)s] [%(name)s] [Stage:%(stage)s] %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(StageContextFilter(stage_id))
    return handler


def _ensure_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    for h in list(logger.handlers):
        try:
            # Deduplicate by handler class and target filename when possible
            if isinstance(h, type(handler)):
                if hasattr(h, "baseFilename") and hasattr(handler, "baseFilename"):
                    if getattr(h, "baseFilename") == getattr(handler, "baseFilename"):
                        return
        except Exception:
            continue
    logger.addHandler(handler)


def configure_stage_logging(stage_id: int, log_file_prefix: Optional[str]) -> None:
    """Attach per-stage file handler to 'vllm' and 'vllm_omni' loggers.

    The handler writes to:
      - {log_file_prefix}.stage{stage_id}.log if log_file_prefix is provided
      - /tmp/omni_stage{stage_id}.log otherwise
    """
    if log_file_prefix:
        file_path = f"{log_file_prefix}.stage{stage_id}.log"
    else:
        file_path = f"/tmp/omni_stage{stage_id}.log"

    file_handler = _build_stage_file_handler(file_path, stage_id)

    # Attach to vLLM and vllm_omni namespaces
    for name in ("vllm", "vllm_omni", "vllm_omni.stage"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        _ensure_handler(lg, file_handler)


