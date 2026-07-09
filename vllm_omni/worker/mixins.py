from __future__ import annotations

from typing import Any


class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    # ISSUE(docstring): missing — add purpose, args, returns, how-it-works
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # the import should be moved to top
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
