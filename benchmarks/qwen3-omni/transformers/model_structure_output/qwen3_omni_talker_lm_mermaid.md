# Qwen3 Omni Model Structure - Talker Language Model

```mermaid
flowchart TD
    talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nLinear\n1x3x2048->1x3x2048"]
    talker_text_projection_act_fn["talker.text_projection.act_fn\nSiLUActivation\n1x3x2048->1x3x2048"]
    talker_text_projection_linear_fc1 --> talker_text_projection_act_fn
    talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nLinear\n1x3x2048->1x3x1024"]
    talker_text_projection_act_fn --> talker_text_projection_linear_fc2
    talker_text_projection["talker.text_projection\nTalkerResizeMLP\n1x3x2048->1x3x1024"]
    talker_text_projection_linear_fc2 --> talker_text_projection
    talker_model_codec_embedding["talker.model.codec_embedding\nEmbedding\n1x6->1x6x1024"]
    talker_text_projection --> talker_model_codec_embedding
    talker_model_rotary_emb["talker.model.rotary_emb\nTalkerRotaryEmbeddin\n1x20x1024->1x20x128"]
    talker_model_codec_embedding --> talker_model_rotary_emb
    talker_model_layers_0_input_layernorm["talker...0.input_layernorm\nThinkerTextRMSNorm\n1x20x1024->1x20x1024"]
    talker_model_rotary_emb --> talker_model_layers_0_input_layernorm
    talker_model_layers_0_self_attn["talker...0.self_attn\nThinkerTextAttention"]
    talker_model_layers_0_input_layernorm --> talker_model_layers_0_self_attn
    talker_model_layers_0_post_attention_layernorm["talker...0.post_attention_layernorm\nThinkerTextRMSNorm\n1x20x1024->1x20x1024"]
    talker_model_layers_0_self_attn --> talker_model_layers_0_post_attention_layernorm
    talker_model_layers_0["talker...layers.0\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
    talker_model_layers_0_post_attention_layernorm --> talker_model_layers_0
    talker_model_norm["talker.model.norm\nTextRMSNorm\n1x20x1024->1x20x1024"]
    talker_model_layers_0 --> talker_model_norm
    talker_model["talker.model\nTalkerModel"]
    talker_model_norm --> talker_model
    talker["talker\nTalkerForConditional"]
    talker_model --> talker
```
