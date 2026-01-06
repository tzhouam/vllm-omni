import glob
import os
import time

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_download_from_modelscope,
)
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


class OmniDefaultModelLoader(DefaultModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads", "download_hook"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected_keys}"
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == "safetensors" or load_format == "fastsafetensors":
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        download_hook = None
        if isinstance(self.load_config.model_loader_extra_config, dict):
            download_hook = self.load_config.model_loader_extra_config.get("download_hook")

        def _emit(event: str, info: dict) -> None:
            if download_hook is None:
                return
            try:
                download_hook(event, info)
            except Exception:
                logger.debug("download_hook failed on event=%s", event)

        if not is_local:
            _emit(
                "download_start",
                {"model": model_name_or_path, "revision": revision},
            )
            _t0 = time.perf_counter()
            try:
                hf_folder = download_weights_from_hf(
                    model_name_or_path,
                    self.load_config.download_dir,
                    allow_patterns,
                    revision,
                    ignore_patterns=self.load_config.ignore_patterns,
                )
            finally:
                _emit(
                    "download_done",
                    {
                        "model": model_name_or_path,
                        "revision": revision,
                        "duration_ms": (time.perf_counter() - _t0) * 1000,
                    },
                )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors
