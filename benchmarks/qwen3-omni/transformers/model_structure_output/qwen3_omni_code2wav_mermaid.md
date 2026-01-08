# Qwen3 Omni Model Structure - Code2wav

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_code2wav_code_embedding["code2wav.code_embedding\nType: Embedding\nIn: 1x16x49 int64\nOut: 1x16x49x1024"]
        P_code2wav_pre_transformer_rotary_emb["code2wav.pre_transformer.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x49x1024 bfloat16\nOut: 1x49x64"]
        P_code2wav_code_embedding --> P_code2wav_pre_transformer_rotary_emb
        P_code2wav_pre_transformer_layers_0_input_layernorm["code2wav.pre_transformer.layers.0.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_rotary_emb --> P_code2wav_pre_transformer_layers_0_input_layernorm
        P_code2wav_pre_transformer_layers_0_self_attn["code2wav.pre_transformer.layers.0.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_0_input_layernorm --> P_code2wav_pre_transformer_layers_0_self_attn
        P_code2wav_pre_transformer_layers_0_self_attn_layer_scale["code2wav.pre_transformer.layers.0.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0_self_attn --> P_code2wav_pre_transformer_layers_0_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_0_post_attention_layernorm["code2wav.pre_transformer.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_0_post_attention_layernorm
        P_code2wav_pre_transformer_layers_0_mlp["code2wav.pre_transformer.layers.0.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0_post_attention_layernorm --> P_code2wav_pre_transformer_layers_0_mlp
        P_code2wav_pre_transformer_layers_0_mlp_layer_scale["code2wav.pre_transformer.layers.0.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0_mlp --> P_code2wav_pre_transformer_layers_0_mlp_layer_scale
        P_code2wav_pre_transformer_layers_0["code2wav.pre_transformer.layers.0\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0_mlp_layer_scale --> P_code2wav_pre_transformer_layers_0
        P_code2wav_pre_transformer_layers_1_input_layernorm["code2wav.pre_transformer.layers.1.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_0 --> P_code2wav_pre_transformer_layers_1_input_layernorm
        P_code2wav_pre_transformer_layers_1_self_attn["code2wav.pre_transformer.layers.1.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_1_input_layernorm --> P_code2wav_pre_transformer_layers_1_self_attn
        P_code2wav_pre_transformer_layers_1_self_attn_layer_scale["code2wav.pre_transformer.layers.1.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1_self_attn --> P_code2wav_pre_transformer_layers_1_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_1_post_attention_layernorm["code2wav.pre_transformer.layers.1.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_1_post_attention_layernorm
        P_code2wav_pre_transformer_layers_1_mlp["code2wav.pre_transformer.layers.1.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1_post_attention_layernorm --> P_code2wav_pre_transformer_layers_1_mlp
        P_code2wav_pre_transformer_layers_1_mlp_layer_scale["code2wav.pre_transformer.layers.1.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1_mlp --> P_code2wav_pre_transformer_layers_1_mlp_layer_scale
        P_code2wav_pre_transformer_layers_1["code2wav.pre_transformer.layers.1\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1_mlp_layer_scale --> P_code2wav_pre_transformer_layers_1
        P_code2wav_pre_transformer_layers_2_input_layernorm["code2wav.pre_transformer.layers.2.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_1 --> P_code2wav_pre_transformer_layers_2_input_layernorm
        P_code2wav_pre_transformer_layers_2_self_attn["code2wav.pre_transformer.layers.2.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_2_input_layernorm --> P_code2wav_pre_transformer_layers_2_self_attn
        P_code2wav_pre_transformer_layers_2_self_attn_layer_scale["code2wav.pre_transformer.layers.2.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2_self_attn --> P_code2wav_pre_transformer_layers_2_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_2_post_attention_layernorm["code2wav.pre_transformer.layers.2.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_2_post_attention_layernorm
        P_code2wav_pre_transformer_layers_2_mlp["code2wav.pre_transformer.layers.2.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2_post_attention_layernorm --> P_code2wav_pre_transformer_layers_2_mlp
        P_code2wav_pre_transformer_layers_2_mlp_layer_scale["code2wav.pre_transformer.layers.2.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2_mlp --> P_code2wav_pre_transformer_layers_2_mlp_layer_scale
        P_code2wav_pre_transformer_layers_2["code2wav.pre_transformer.layers.2\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2_mlp_layer_scale --> P_code2wav_pre_transformer_layers_2
        P_code2wav_pre_transformer_layers_3_input_layernorm["code2wav.pre_transformer.layers.3.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_2 --> P_code2wav_pre_transformer_layers_3_input_layernorm
        P_code2wav_pre_transformer_layers_3_self_attn["code2wav.pre_transformer.layers.3.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_3_input_layernorm --> P_code2wav_pre_transformer_layers_3_self_attn
        P_code2wav_pre_transformer_layers_3_self_attn_layer_scale["code2wav.pre_transformer.layers.3.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3_self_attn --> P_code2wav_pre_transformer_layers_3_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_3_post_attention_layernorm["code2wav.pre_transformer.layers.3.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_3_post_attention_layernorm
        P_code2wav_pre_transformer_layers_3_mlp["code2wav.pre_transformer.layers.3.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3_post_attention_layernorm --> P_code2wav_pre_transformer_layers_3_mlp
        P_code2wav_pre_transformer_layers_3_mlp_layer_scale["code2wav.pre_transformer.layers.3.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3_mlp --> P_code2wav_pre_transformer_layers_3_mlp_layer_scale
        P_code2wav_pre_transformer_layers_3["code2wav.pre_transformer.layers.3\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3_mlp_layer_scale --> P_code2wav_pre_transformer_layers_3
        P_code2wav_pre_transformer_layers_4_input_layernorm["code2wav.pre_transformer.layers.4.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_3 --> P_code2wav_pre_transformer_layers_4_input_layernorm
        P_code2wav_pre_transformer_layers_4_self_attn["code2wav.pre_transformer.layers.4.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_4_input_layernorm --> P_code2wav_pre_transformer_layers_4_self_attn
        P_code2wav_pre_transformer_layers_4_self_attn_layer_scale["code2wav.pre_transformer.layers.4.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4_self_attn --> P_code2wav_pre_transformer_layers_4_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_4_post_attention_layernorm["code2wav.pre_transformer.layers.4.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_4_post_attention_layernorm
        P_code2wav_pre_transformer_layers_4_mlp["code2wav.pre_transformer.layers.4.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4_post_attention_layernorm --> P_code2wav_pre_transformer_layers_4_mlp
        P_code2wav_pre_transformer_layers_4_mlp_layer_scale["code2wav.pre_transformer.layers.4.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4_mlp --> P_code2wav_pre_transformer_layers_4_mlp_layer_scale
        P_code2wav_pre_transformer_layers_4["code2wav.pre_transformer.layers.4\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4_mlp_layer_scale --> P_code2wav_pre_transformer_layers_4
        P_code2wav_pre_transformer_layers_5_input_layernorm["code2wav.pre_transformer.layers.5.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_4 --> P_code2wav_pre_transformer_layers_5_input_layernorm
        P_code2wav_pre_transformer_layers_5_self_attn["code2wav.pre_transformer.layers.5.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_5_input_layernorm --> P_code2wav_pre_transformer_layers_5_self_attn
        P_code2wav_pre_transformer_layers_5_self_attn_layer_scale["code2wav.pre_transformer.layers.5.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5_self_attn --> P_code2wav_pre_transformer_layers_5_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_5_post_attention_layernorm["code2wav.pre_transformer.layers.5.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_5_post_attention_layernorm
        P_code2wav_pre_transformer_layers_5_mlp["code2wav.pre_transformer.layers.5.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5_post_attention_layernorm --> P_code2wav_pre_transformer_layers_5_mlp
        P_code2wav_pre_transformer_layers_5_mlp_layer_scale["code2wav.pre_transformer.layers.5.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5_mlp --> P_code2wav_pre_transformer_layers_5_mlp_layer_scale
        P_code2wav_pre_transformer_layers_5["code2wav.pre_transformer.layers.5\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5_mlp_layer_scale --> P_code2wav_pre_transformer_layers_5
        P_code2wav_pre_transformer_layers_6_input_layernorm["code2wav.pre_transformer.layers.6.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_5 --> P_code2wav_pre_transformer_layers_6_input_layernorm
        P_code2wav_pre_transformer_layers_6_self_attn["code2wav.pre_transformer.layers.6.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_6_input_layernorm --> P_code2wav_pre_transformer_layers_6_self_attn
        P_code2wav_pre_transformer_layers_6_self_attn_layer_scale["code2wav.pre_transformer.layers.6.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6_self_attn --> P_code2wav_pre_transformer_layers_6_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_6_post_attention_layernorm["code2wav.pre_transformer.layers.6.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_6_post_attention_layernorm
        P_code2wav_pre_transformer_layers_6_mlp["code2wav.pre_transformer.layers.6.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6_post_attention_layernorm --> P_code2wav_pre_transformer_layers_6_mlp
        P_code2wav_pre_transformer_layers_6_mlp_layer_scale["code2wav.pre_transformer.layers.6.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6_mlp --> P_code2wav_pre_transformer_layers_6_mlp_layer_scale
        P_code2wav_pre_transformer_layers_6["code2wav.pre_transformer.layers.6\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6_mlp_layer_scale --> P_code2wav_pre_transformer_layers_6
        P_code2wav_pre_transformer_layers_7_input_layernorm["code2wav.pre_transformer.layers.7.input_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_6 --> P_code2wav_pre_transformer_layers_7_input_layernorm
        P_code2wav_pre_transformer_layers_7_self_attn["code2wav.pre_transformer.layers.7.self_attn\nQwen3OmniMoeCode2WavAttention"]
        P_code2wav_pre_transformer_layers_7_input_layernorm --> P_code2wav_pre_transformer_layers_7_self_attn
        P_code2wav_pre_transformer_layers_7_self_attn_layer_scale["code2wav.pre_transformer.layers.7.self_attn_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7_self_attn --> P_code2wav_pre_transformer_layers_7_self_attn_layer_scale
        P_code2wav_pre_transformer_layers_7_post_attention_layernorm["code2wav.pre_transformer.layers.7.post_attention_layernorm\nType: Qwen3OmniMoeCode2WavRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7_self_attn_layer_scale --> P_code2wav_pre_transformer_layers_7_post_attention_layernorm
        P_code2wav_pre_transformer_layers_7_mlp["code2wav.pre_transformer.layers.7.mlp\nType: Qwen3OmniMoeCode2WavMlp\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7_post_attention_layernorm --> P_code2wav_pre_transformer_layers_7_mlp
        P_code2wav_pre_transformer_layers_7_mlp_layer_scale["code2wav.pre_transformer.layers.7.mlp_layer_scale\nType: Qwen3OmniMoeCode2WavLayerScale\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7_mlp --> P_code2wav_pre_transformer_layers_7_mlp_layer_scale
        P_code2wav_pre_transformer_layers_7["code2wav.pre_transformer.layers.7\nType: Qwen3OmniMoeCode2WavTransformerLayer\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7_mlp_layer_scale --> P_code2wav_pre_transformer_layers_7
        P_code2wav_pre_transformer_norm["code2wav.pre_transformer.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x49x1024 bfloat16\nOut: 1x49x1024"]
        P_code2wav_pre_transformer_layers_7 --> P_code2wav_pre_transformer_norm
        P_code2wav_pre_transformer["code2wav.pre_transformer\nQwen3OmniMoeCode2WavTransformerModel"]
        P_code2wav_pre_transformer_norm --> P_code2wav_pre_transformer
        P_code2wav_upsample_0_0_conv["code2wav.upsample.0.0.conv\nType: ConvTranspose1d\nIn: 1x1024x49 bfloat16\nOut: 1x1024x98"]
        P_code2wav_pre_transformer --> P_code2wav_upsample_0_0_conv
        P_code2wav_upsample_0_0["code2wav.upsample.0.0\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x1024x49 bfloat16\nOut: 1x1024x98"]
        P_code2wav_upsample_0_0_conv --> P_code2wav_upsample_0_0
        P_code2wav_upsample_0_1_dwconv["code2wav.upsample.0.1.dwconv\nType: Qwen3OmniMoeCausalConvNet\nIn: 1x1024x98 bfloat16\nOut: 1x1024x98"]
        P_code2wav_upsample_0_0 --> P_code2wav_upsample_0_1_dwconv
        P_code2wav_upsample_0_1_norm["code2wav.upsample.0.1.norm\nType: LayerNorm\nIn: 1x98x1024 bfloat16\nOut: 1x98x1024"]
        P_code2wav_upsample_0_1_dwconv --> P_code2wav_upsample_0_1_norm
        P_code2wav_upsample_0_1_pwconv1["code2wav.upsample.0.1.pwconv1\nType: Linear\nIn: 1x98x1024 bfloat16\nOut: 1x98x4096"]
        P_code2wav_upsample_0_1_norm --> P_code2wav_upsample_0_1_pwconv1
        P_code2wav_upsample_0_1_act["code2wav.upsample.0.1.act\nType: GELU\nIn: 1x98x4096 bfloat16\nOut: 1x98x4096"]
        P_code2wav_upsample_0_1_pwconv1 --> P_code2wav_upsample_0_1_act
        P_code2wav_upsample_0_1_pwconv2["code2wav.upsample.0.1.pwconv2\nType: Linear\nIn: 1x98x4096 bfloat16\nOut: 1x98x1024"]
        P_code2wav_upsample_0_1_act --> P_code2wav_upsample_0_1_pwconv2
        P_code2wav_upsample_0_1["code2wav.upsample.0.1\nType: Qwen3OmniMoeConvNeXtBlock\nIn: 1x1024x98 bfloat16\nOut: 1x1024x98"]
        P_code2wav_upsample_0_1_pwconv2 --> P_code2wav_upsample_0_1
        P_code2wav_upsample_1_0_conv["code2wav.upsample.1.0.conv\nType: ConvTranspose1d\nIn: 1x1024x98 bfloat16\nOut: 1x1024x196"]
        P_code2wav_upsample_0_1 --> P_code2wav_upsample_1_0_conv
        P_code2wav_upsample_1_0["code2wav.upsample.1.0\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x1024x98 bfloat16\nOut: 1x1024x196"]
        P_code2wav_upsample_1_0_conv --> P_code2wav_upsample_1_0
        P_code2wav_upsample_1_1_dwconv["code2wav.upsample.1.1.dwconv\nType: Qwen3OmniMoeCausalConvNet\nIn: 1x1024x196 bfloat16\nOut: 1x1024x196"]
        P_code2wav_upsample_1_0 --> P_code2wav_upsample_1_1_dwconv
        P_code2wav_upsample_1_1_norm["code2wav.upsample.1.1.norm\nType: LayerNorm\nIn: 1x196x1024 bfloat16\nOut: 1x196x1024"]
        P_code2wav_upsample_1_1_dwconv --> P_code2wav_upsample_1_1_norm
        P_code2wav_upsample_1_1_pwconv1["code2wav.upsample.1.1.pwconv1\nType: Linear\nIn: 1x196x1024 bfloat16\nOut: 1x196x4096"]
        P_code2wav_upsample_1_1_norm --> P_code2wav_upsample_1_1_pwconv1
        P_code2wav_upsample_1_1_act["code2wav.upsample.1.1.act\nType: GELU\nIn: 1x196x4096 bfloat16\nOut: 1x196x4096"]
        P_code2wav_upsample_1_1_pwconv1 --> P_code2wav_upsample_1_1_act
        P_code2wav_upsample_1_1_pwconv2["code2wav.upsample.1.1.pwconv2\nType: Linear\nIn: 1x196x4096 bfloat16\nOut: 1x196x1024"]
        P_code2wav_upsample_1_1_act --> P_code2wav_upsample_1_1_pwconv2
        P_code2wav_upsample_1_1["code2wav.upsample.1.1\nType: Qwen3OmniMoeConvNeXtBlock\nIn: 1x1024x196 bfloat16\nOut: 1x1024x196"]
        P_code2wav_upsample_1_1_pwconv2 --> P_code2wav_upsample_1_1
        P_code2wav_decoder_0_conv["code2wav.decoder.0.conv\nType: Conv1d\nIn: 1x1024x202 bfloat16\nOut: 1x1536x196"]
        P_code2wav_upsample_1_1 --> P_code2wav_decoder_0_conv
        P_code2wav_decoder_0["code2wav.decoder.0\nType: Qwen3OmniMoeCausalConvNet\nIn: 1x1024x196 bfloat16\nOut: 1x1536x196"]
        P_code2wav_decoder_0_conv --> P_code2wav_decoder_0
        P_code2wav_decoder_1_block_0["code2wav.decoder.1.block.0\nType: SnakeBeta\nIn: 1x1536x196 bfloat16\nOut: 1x1536x196"]
        P_code2wav_decoder_0 --> P_code2wav_decoder_1_block_0
        P_code2wav_decoder_1_block_1["code2wav.decoder.1.block.1\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x1536x196 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1_block_0 --> P_code2wav_decoder_1_block_1
        P_code2wav_decoder_1_block_2["code2wav.decoder.1.block.2\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x768x1560 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1_block_1 --> P_code2wav_decoder_1_block_2
        P_code2wav_decoder_1_block_3["code2wav.decoder.1.block.3\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x768x1560 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1_block_2 --> P_code2wav_decoder_1_block_3
        P_code2wav_decoder_1_block_4["code2wav.decoder.1.block.4\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x768x1560 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1_block_3 --> P_code2wav_decoder_1_block_4
        P_code2wav_decoder_1["code2wav.decoder.1\nType: Qwen3OmniMoeCode2WavDecoderBlock\nIn: 1x1536x196 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1_block_4 --> P_code2wav_decoder_1
        P_code2wav_decoder_2_block_0["code2wav.decoder.2.block.0\nType: SnakeBeta\nIn: 1x768x1560 bfloat16\nOut: 1x768x1560"]
        P_code2wav_decoder_1 --> P_code2wav_decoder_2_block_0
        P_code2wav_decoder_2_block_1["code2wav.decoder.2.block.1\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x768x1560 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2_block_0 --> P_code2wav_decoder_2_block_1
        P_code2wav_decoder_2_block_2["code2wav.decoder.2.block.2\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x384x7795 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2_block_1 --> P_code2wav_decoder_2_block_2
        P_code2wav_decoder_2_block_3["code2wav.decoder.2.block.3\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x384x7795 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2_block_2 --> P_code2wav_decoder_2_block_3
        P_code2wav_decoder_2_block_4["code2wav.decoder.2.block.4\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x384x7795 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2_block_3 --> P_code2wav_decoder_2_block_4
        P_code2wav_decoder_2["code2wav.decoder.2\nType: Qwen3OmniMoeCode2WavDecoderBlock\nIn: 1x768x1560 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2_block_4 --> P_code2wav_decoder_2
        P_code2wav_decoder_3_block_0["code2wav.decoder.3.block.0\nType: SnakeBeta\nIn: 1x384x7795 bfloat16\nOut: 1x384x7795"]
        P_code2wav_decoder_2 --> P_code2wav_decoder_3_block_0
        P_code2wav_decoder_3_block_1["code2wav.decoder.3.block.1\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x384x7795 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3_block_0 --> P_code2wav_decoder_3_block_1
        P_code2wav_decoder_3_block_2["code2wav.decoder.3.block.2\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x192x31176 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3_block_1 --> P_code2wav_decoder_3_block_2
        P_code2wav_decoder_3_block_3["code2wav.decoder.3.block.3\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x192x31176 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3_block_2 --> P_code2wav_decoder_3_block_3
        P_code2wav_decoder_3_block_4["code2wav.decoder.3.block.4\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x192x31176 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3_block_3 --> P_code2wav_decoder_3_block_4
        P_code2wav_decoder_3["code2wav.decoder.3\nType: Qwen3OmniMoeCode2WavDecoderBlock\nIn: 1x384x7795 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3_block_4 --> P_code2wav_decoder_3
        P_code2wav_decoder_4_block_0["code2wav.decoder.4.block.0\nType: SnakeBeta\nIn: 1x192x31176 bfloat16\nOut: 1x192x31176"]
        P_code2wav_decoder_3 --> P_code2wav_decoder_4_block_0
        P_code2wav_decoder_4_block_1["code2wav.decoder.4.block.1\nType: Qwen3OmniMoeCausalTransConvNet\nIn: 1x192x31176 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4_block_0 --> P_code2wav_decoder_4_block_1
        P_code2wav_decoder_4_block_2["code2wav.decoder.4.block.2\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x96x93525 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4_block_1 --> P_code2wav_decoder_4_block_2
        P_code2wav_decoder_4_block_3["code2wav.decoder.4.block.3\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x96x93525 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4_block_2 --> P_code2wav_decoder_4_block_3
        P_code2wav_decoder_4_block_4["code2wav.decoder.4.block.4\nType: Qwen3OmniMoeCode2WavDecoderResidualUnit\nIn: 1x96x93525 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4_block_3 --> P_code2wav_decoder_4_block_4
        P_code2wav_decoder_4["code2wav.decoder.4\nType: Qwen3OmniMoeCode2WavDecoderBlock\nIn: 1x192x31176 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4_block_4 --> P_code2wav_decoder_4
        P_code2wav_decoder_5["code2wav.decoder.5\nType: SnakeBeta\nIn: 1x96x93525 bfloat16\nOut: 1x96x93525"]
        P_code2wav_decoder_4 --> P_code2wav_decoder_5
        P_code2wav_decoder_6_conv["code2wav.decoder.6.conv\nType: Conv1d\nIn: 1x96x93531 bfloat16\nOut: 1x1x93525"]
        P_code2wav_decoder_5 --> P_code2wav_decoder_6_conv
        P_code2wav_decoder_6["code2wav.decoder.6\nType: Qwen3OmniMoeCausalConvNet\nIn: 1x96x93525 bfloat16\nOut: 1x1x93525"]
        P_code2wav_decoder_6_conv --> P_code2wav_decoder_6
        P_code2wav["code2wav\nType: Qwen3OmniMoeCode2Wav\nIn: 1x16x49 int64\nOut: 1x1x93525"]
        P_code2wav_decoder_6 --> P_code2wav
    end
```
