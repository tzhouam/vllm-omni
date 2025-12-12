"""Thin Omni wrapper: reuse upstream Qwen2.5-Omni thinker (v0.12) with minimal overrides."""

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniThinker,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalDataParser,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
    create_qwen2_5_omni_thinker_field_factory,
)

# Re-export upstream classes with Omni aliases for compatibility
class Qwen2_5OmniThinkerMultiModalDataParser(Qwen2_5OmniThinkerMultiModalDataParser):
    pass


class Qwen2_5OmniThinkerProcessingInfo(Qwen2_5OmniThinkerProcessingInfo):
    pass


class Qwen2_5OmniThinkerDummyInputsBuilder(Qwen2_5OmniThinkerDummyInputsBuilder):
    pass


class Qwen2_5OmniThinkerMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    pass


class Qwen2_5OmniThinker(Qwen2_5OmniThinker):
    pass
