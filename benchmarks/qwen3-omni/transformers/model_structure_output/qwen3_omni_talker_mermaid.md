# Qwen3 Omni Model Structure - Talker

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nType: Linear\nIn: 1x3x2048 bfloat16\nOut: 1x3x2048"]
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nType: SiLUActivation\nIn: 1x3x2048 bfloat16\nOut: 1x3x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nType: Linear\nIn: 1x3x2048 bfloat16\nOut: 1x3x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nType: Qwen3OmniMoeTalkerResizeMLP\nIn: 1x3x2048 bfloat16\nOut: 1x3x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nType: Linear\nIn: 11x2048 bfloat16\nOut: 11x2048"]
        P_talker_text_projection --> P_talker_text_projection_linear_fc1
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nType: SiLUActivation\nIn: 11x2048 bfloat16\nOut: 11x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nType: Linear\nIn: 11x2048 bfloat16\nOut: 11x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nType: Qwen3OmniMoeTalkerResizeMLP\nIn: 11x2048 bfloat16\nOut: 11x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nType: Linear\nIn: 1x20x2048 bfloat16\nOut: 1x20x2048"]
        P_talker_text_projection --> P_talker_text_projection_linear_fc1
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nType: SiLUActivation\nIn: 1x20x2048 bfloat16\nOut: 1x20x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nType: Linear\nIn: 1x20x2048 bfloat16\nOut: 1x20x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nType: Qwen3OmniMoeTalkerResizeMLP\nIn: 1x20x2048 bfloat16\nOut: 1x20x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_model_codec_embedding["talker.model.codec_embedding\nType: Embedding\nIn: 1x6 int64\nOut: 1x6x1024"]
        P_talker_text_projection --> P_talker_model_codec_embedding
        P_talker_model_rotary_emb["talker.model.rotary_emb\nType: Qwen3OmniMoeTalkerRotaryEmbedding\nIn: 1x20x1024 bfloat16\nOut: 1x20x128"]
        P_talker_model_codec_embedding --> P_talker_model_rotary_emb
        P_talker_model_layers_0_input_layernorm["talker.model.layers.0.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_rotary_emb --> P_talker_model_layers_0_input_layernorm
        P_talker_model_layers_0_self_attn["talker.model.layers.0.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_0_input_layernorm --> P_talker_model_layers_0_self_attn
        P_talker_model_layers_0_post_attention_layernorm["talker.model.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_0_self_attn --> P_talker_model_layers_0_post_attention_layernorm
        P_talker_model_layers_0_mlp["talker.model.layers.0.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_0_post_attention_layernorm --> P_talker_model_layers_0_mlp
        P_talker_model_layers_0["talker.model.layers.0\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_0_mlp --> P_talker_model_layers_0
        P_talker_model_layers_1_input_layernorm["talker.model.layers.1.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_0 --> P_talker_model_layers_1_input_layernorm
        P_talker_model_layers_1_self_attn["talker.model.layers.1.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_1_input_layernorm --> P_talker_model_layers_1_self_attn
        P_talker_model_layers_1_post_attention_layernorm["talker.model.layers.1.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_1_self_attn --> P_talker_model_layers_1_post_attention_layernorm
        P_talker_model_layers_1_mlp["talker.model.layers.1.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_1_post_attention_layernorm --> P_talker_model_layers_1_mlp
        P_talker_model_layers_1["talker.model.layers.1\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_1_mlp --> P_talker_model_layers_1
        P_talker_model_layers_2_input_layernorm["talker.model.layers.2.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_1 --> P_talker_model_layers_2_input_layernorm
        P_talker_model_layers_2_self_attn["talker.model.layers.2.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_2_input_layernorm --> P_talker_model_layers_2_self_attn
        P_talker_model_layers_2_post_attention_layernorm["talker.model.layers.2.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_2_self_attn --> P_talker_model_layers_2_post_attention_layernorm
        P_talker_model_layers_2_mlp["talker.model.layers.2.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_2_post_attention_layernorm --> P_talker_model_layers_2_mlp
        P_talker_model_layers_2["talker.model.layers.2\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_2_mlp --> P_talker_model_layers_2
        P_talker_model_layers_3_input_layernorm["talker.model.layers.3.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_2 --> P_talker_model_layers_3_input_layernorm
        P_talker_model_layers_3_self_attn["talker.model.layers.3.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_3_input_layernorm --> P_talker_model_layers_3_self_attn
        P_talker_model_layers_3_post_attention_layernorm["talker.model.layers.3.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_3_self_attn --> P_talker_model_layers_3_post_attention_layernorm
        P_talker_model_layers_3_mlp["talker.model.layers.3.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_3_post_attention_layernorm --> P_talker_model_layers_3_mlp
        P_talker_model_layers_3["talker.model.layers.3\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_3_mlp --> P_talker_model_layers_3
        P_talker_model_layers_4_input_layernorm["talker.model.layers.4.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_3 --> P_talker_model_layers_4_input_layernorm
        P_talker_model_layers_4_self_attn["talker.model.layers.4.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_4_input_layernorm --> P_talker_model_layers_4_self_attn
        P_talker_model_layers_4_post_attention_layernorm["talker.model.layers.4.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_4_self_attn --> P_talker_model_layers_4_post_attention_layernorm
        P_talker_model_layers_4_mlp["talker.model.layers.4.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_4_post_attention_layernorm --> P_talker_model_layers_4_mlp
        P_talker_model_layers_4["talker.model.layers.4\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_4_mlp --> P_talker_model_layers_4
        P_talker_model_layers_5_input_layernorm["talker.model.layers.5.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_4 --> P_talker_model_layers_5_input_layernorm
        P_talker_model_layers_5_self_attn["talker.model.layers.5.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_5_input_layernorm --> P_talker_model_layers_5_self_attn
        P_talker_model_layers_5_post_attention_layernorm["talker.model.layers.5.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_5_self_attn --> P_talker_model_layers_5_post_attention_layernorm
        P_talker_model_layers_5_mlp["talker.model.layers.5.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_5_post_attention_layernorm --> P_talker_model_layers_5_mlp
        P_talker_model_layers_5["talker.model.layers.5\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_5_mlp --> P_talker_model_layers_5
        P_talker_model_layers_6_input_layernorm["talker.model.layers.6.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_5 --> P_talker_model_layers_6_input_layernorm
        P_talker_model_layers_6_self_attn["talker.model.layers.6.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_6_input_layernorm --> P_talker_model_layers_6_self_attn
        P_talker_model_layers_6_post_attention_layernorm["talker.model.layers.6.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_6_self_attn --> P_talker_model_layers_6_post_attention_layernorm
        P_talker_model_layers_6_mlp["talker.model.layers.6.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_6_post_attention_layernorm --> P_talker_model_layers_6_mlp
        P_talker_model_layers_6["talker.model.layers.6\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_6_mlp --> P_talker_model_layers_6
        P_talker_model_layers_7_input_layernorm["talker.model.layers.7.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_6 --> P_talker_model_layers_7_input_layernorm
        P_talker_model_layers_7_self_attn["talker.model.layers.7.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_7_input_layernorm --> P_talker_model_layers_7_self_attn
        P_talker_model_layers_7_post_attention_layernorm["talker.model.layers.7.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_7_self_attn --> P_talker_model_layers_7_post_attention_layernorm
        P_talker_model_layers_7_mlp["talker.model.layers.7.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_7_post_attention_layernorm --> P_talker_model_layers_7_mlp
        P_talker_model_layers_7["talker.model.layers.7\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_7_mlp --> P_talker_model_layers_7
        P_talker_model_layers_8_input_layernorm["talker.model.layers.8.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_7 --> P_talker_model_layers_8_input_layernorm
        P_talker_model_layers_8_self_attn["talker.model.layers.8.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_8_input_layernorm --> P_talker_model_layers_8_self_attn
        P_talker_model_layers_8_post_attention_layernorm["talker.model.layers.8.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_8_self_attn --> P_talker_model_layers_8_post_attention_layernorm
        P_talker_model_layers_8_mlp["talker.model.layers.8.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_8_post_attention_layernorm --> P_talker_model_layers_8_mlp
        P_talker_model_layers_8["talker.model.layers.8\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_8_mlp --> P_talker_model_layers_8
        P_talker_model_layers_9_input_layernorm["talker.model.layers.9.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_8 --> P_talker_model_layers_9_input_layernorm
        P_talker_model_layers_9_self_attn["talker.model.layers.9.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_9_input_layernorm --> P_talker_model_layers_9_self_attn
        P_talker_model_layers_9_post_attention_layernorm["talker.model.layers.9.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_9_self_attn --> P_talker_model_layers_9_post_attention_layernorm
        P_talker_model_layers_9_mlp["talker.model.layers.9.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_9_post_attention_layernorm --> P_talker_model_layers_9_mlp
        P_talker_model_layers_9["talker.model.layers.9\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_9_mlp --> P_talker_model_layers_9
        P_talker_model_layers_10_input_layernorm["talker.model.layers.10.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_9 --> P_talker_model_layers_10_input_layernorm
        P_talker_model_layers_10_self_attn["talker.model.layers.10.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_10_input_layernorm --> P_talker_model_layers_10_self_attn
        P_talker_model_layers_10_post_attention_layernorm["talker.model.layers.10.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_10_self_attn --> P_talker_model_layers_10_post_attention_layernorm
        P_talker_model_layers_10_mlp["talker.model.layers.10.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_10_post_attention_layernorm --> P_talker_model_layers_10_mlp
        P_talker_model_layers_10["talker.model.layers.10\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_10_mlp --> P_talker_model_layers_10
        P_talker_model_layers_11_input_layernorm["talker.model.layers.11.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_10 --> P_talker_model_layers_11_input_layernorm
        P_talker_model_layers_11_self_attn["talker.model.layers.11.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_11_input_layernorm --> P_talker_model_layers_11_self_attn
        P_talker_model_layers_11_post_attention_layernorm["talker.model.layers.11.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_11_self_attn --> P_talker_model_layers_11_post_attention_layernorm
        P_talker_model_layers_11_mlp["talker.model.layers.11.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_11_post_attention_layernorm --> P_talker_model_layers_11_mlp
        P_talker_model_layers_11["talker.model.layers.11\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_11_mlp --> P_talker_model_layers_11
        P_talker_model_layers_12_input_layernorm["talker.model.layers.12.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_11 --> P_talker_model_layers_12_input_layernorm
        P_talker_model_layers_12_self_attn["talker.model.layers.12.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_12_input_layernorm --> P_talker_model_layers_12_self_attn
        P_talker_model_layers_12_post_attention_layernorm["talker.model.layers.12.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_12_self_attn --> P_talker_model_layers_12_post_attention_layernorm
        P_talker_model_layers_12_mlp["talker.model.layers.12.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_12_post_attention_layernorm --> P_talker_model_layers_12_mlp
        P_talker_model_layers_12["talker.model.layers.12\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_12_mlp --> P_talker_model_layers_12
        P_talker_model_layers_13_input_layernorm["talker.model.layers.13.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_12 --> P_talker_model_layers_13_input_layernorm
        P_talker_model_layers_13_self_attn["talker.model.layers.13.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_13_input_layernorm --> P_talker_model_layers_13_self_attn
        P_talker_model_layers_13_post_attention_layernorm["talker.model.layers.13.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_13_self_attn --> P_talker_model_layers_13_post_attention_layernorm
        P_talker_model_layers_13_mlp["talker.model.layers.13.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_13_post_attention_layernorm --> P_talker_model_layers_13_mlp
        P_talker_model_layers_13["talker.model.layers.13\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_13_mlp --> P_talker_model_layers_13
        P_talker_model_layers_14_input_layernorm["talker.model.layers.14.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_13 --> P_talker_model_layers_14_input_layernorm
        P_talker_model_layers_14_self_attn["talker.model.layers.14.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_14_input_layernorm --> P_talker_model_layers_14_self_attn
        P_talker_model_layers_14_post_attention_layernorm["talker.model.layers.14.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_14_self_attn --> P_talker_model_layers_14_post_attention_layernorm
        P_talker_model_layers_14_mlp["talker.model.layers.14.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_14_post_attention_layernorm --> P_talker_model_layers_14_mlp
        P_talker_model_layers_14["talker.model.layers.14\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_14_mlp --> P_talker_model_layers_14
        P_talker_model_layers_15_input_layernorm["talker.model.layers.15.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_14 --> P_talker_model_layers_15_input_layernorm
        P_talker_model_layers_15_self_attn["talker.model.layers.15.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_15_input_layernorm --> P_talker_model_layers_15_self_attn
        P_talker_model_layers_15_post_attention_layernorm["talker.model.layers.15.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_15_self_attn --> P_talker_model_layers_15_post_attention_layernorm
        P_talker_model_layers_15_mlp["talker.model.layers.15.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_15_post_attention_layernorm --> P_talker_model_layers_15_mlp
        P_talker_model_layers_15["talker.model.layers.15\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_15_mlp --> P_talker_model_layers_15
        P_talker_model_layers_16_input_layernorm["talker.model.layers.16.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_15 --> P_talker_model_layers_16_input_layernorm
        P_talker_model_layers_16_self_attn["talker.model.layers.16.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_16_input_layernorm --> P_talker_model_layers_16_self_attn
        P_talker_model_layers_16_post_attention_layernorm["talker.model.layers.16.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_16_self_attn --> P_talker_model_layers_16_post_attention_layernorm
        P_talker_model_layers_16_mlp["talker.model.layers.16.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_16_post_attention_layernorm --> P_talker_model_layers_16_mlp
        P_talker_model_layers_16["talker.model.layers.16\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_16_mlp --> P_talker_model_layers_16
        P_talker_model_layers_17_input_layernorm["talker.model.layers.17.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_16 --> P_talker_model_layers_17_input_layernorm
        P_talker_model_layers_17_self_attn["talker.model.layers.17.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_17_input_layernorm --> P_talker_model_layers_17_self_attn
        P_talker_model_layers_17_post_attention_layernorm["talker.model.layers.17.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_17_self_attn --> P_talker_model_layers_17_post_attention_layernorm
        P_talker_model_layers_17_mlp["talker.model.layers.17.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_17_post_attention_layernorm --> P_talker_model_layers_17_mlp
        P_talker_model_layers_17["talker.model.layers.17\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_17_mlp --> P_talker_model_layers_17
        P_talker_model_layers_18_input_layernorm["talker.model.layers.18.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_17 --> P_talker_model_layers_18_input_layernorm
        P_talker_model_layers_18_self_attn["talker.model.layers.18.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_18_input_layernorm --> P_talker_model_layers_18_self_attn
        P_talker_model_layers_18_post_attention_layernorm["talker.model.layers.18.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_18_self_attn --> P_talker_model_layers_18_post_attention_layernorm
        P_talker_model_layers_18_mlp["talker.model.layers.18.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_18_post_attention_layernorm --> P_talker_model_layers_18_mlp
        P_talker_model_layers_18["talker.model.layers.18\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_18_mlp --> P_talker_model_layers_18
        P_talker_model_layers_19_input_layernorm["talker.model.layers.19.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_18 --> P_talker_model_layers_19_input_layernorm
        P_talker_model_layers_19_self_attn["talker.model.layers.19.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_19_input_layernorm --> P_talker_model_layers_19_self_attn
        P_talker_model_layers_19_post_attention_layernorm["talker.model.layers.19.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_19_self_attn --> P_talker_model_layers_19_post_attention_layernorm
        P_talker_model_layers_19_mlp["talker.model.layers.19.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_19_post_attention_layernorm --> P_talker_model_layers_19_mlp
        P_talker_model_layers_19["talker.model.layers.19\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_19_mlp --> P_talker_model_layers_19
        P_talker_model_norm["talker.model.norm\nType: Qwen3OmniMoeTextRMSNorm\nIn: 1x20x1024 bfloat16\nOut: 1x20x1024"]
        P_talker_model_layers_19 --> P_talker_model_norm
        P_talker_model["talker.model\nQwen3OmniMoeTalkerModel"]
        P_talker_model_norm --> P_talker_model
        P_talker_codec_head["talker.codec_head\nType: Linear\nIn: 1x20x1024 bfloat16\nOut: 1x20x3072"]
        P_talker_model --> P_talker_codec_head
        P_talker["talker\nQwen3OmniMoeTalkerForConditionalGeneration"]
        P_talker_codec_head --> P_talker
        P_talker_model_codec_embedding["talker.model.codec_embedding\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker --> P_talker_model_codec_embedding
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x2x1024 bfloat16\nOut: 1x2x128"]
        P_talker_model_codec_embedding --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_0["talker.code_predictor.lm_head.0\nType: Linear\nIn: 1x2x1024 bfloat16\nOut: 1x2x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_0
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_0 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_0["talker.code_predictor.model.codec_embedding.0\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_0
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_0 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_1["talker.code_predictor.lm_head.1\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_1
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_1 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_1["talker.code_predictor.model.codec_embedding.1\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_1
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_1 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_2["talker.code_predictor.lm_head.2\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_2
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_2 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_2["talker.code_predictor.model.codec_embedding.2\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_2
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_2 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_3["talker.code_predictor.lm_head.3\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_3
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_3 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_3["talker.code_predictor.model.codec_embedding.3\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_3
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_3 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_4["talker.code_predictor.lm_head.4\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_4
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_4 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_4["talker.code_predictor.model.codec_embedding.4\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_4
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_4 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_5["talker.code_predictor.lm_head.5\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_5
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_5 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_5["talker.code_predictor.model.codec_embedding.5\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_5
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_5 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_6["talker.code_predictor.lm_head.6\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_6
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_6 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_6["talker.code_predictor.model.codec_embedding.6\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_6
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_6 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_7["talker.code_predictor.lm_head.7\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_7
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_7 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_7["talker.code_predictor.model.codec_embedding.7\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_7
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_7 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_8["talker.code_predictor.lm_head.8\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_8
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_8 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_8["talker.code_predictor.model.codec_embedding.8\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_8
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_8 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_9["talker.code_predictor.lm_head.9\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_9
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_9 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_9["talker.code_predictor.model.codec_embedding.9\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_9
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_9 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_10["talker.code_predictor.lm_head.10\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_10
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_10 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_10["talker.code_predictor.model.codec_embedding.10\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_10
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_10 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_11["talker.code_predictor.lm_head.11\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_11
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_11 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_11["talker.code_predictor.model.codec_embedding.11\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_11
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_11 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_12["talker.code_predictor.lm_head.12\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_12
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_12 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_12["talker.code_predictor.model.codec_embedding.12\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_12
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_12 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_13["talker.code_predictor.lm_head.13\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_13
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_13 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_13["talker.code_predictor.model.codec_embedding.13\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_13
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_13 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_14["talker.code_predictor.lm_head.14\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_14
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_14 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_14["talker.code_predictor.model.codec_embedding.14\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_14
        P_talker_model_rotary_emb["talker.model.rotary_emb\nType: Qwen3OmniMoeTalkerRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_14 --> P_talker_model_rotary_emb
        P_talker_model_layers_0_input_layernorm["talker.model.layers.0.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_rotary_emb --> P_talker_model_layers_0_input_layernorm
        P_talker_model_layers_0_self_attn["talker.model.layers.0.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_0_input_layernorm --> P_talker_model_layers_0_self_attn
        P_talker_model_layers_0_post_attention_layernorm["talker.model.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_0_self_attn --> P_talker_model_layers_0_post_attention_layernorm
        P_talker_model_layers_0_mlp["talker.model.layers.0.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_0_post_attention_layernorm --> P_talker_model_layers_0_mlp
        P_talker_model_layers_0["talker.model.layers.0\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_0_mlp --> P_talker_model_layers_0
        P_talker_model_layers_1_input_layernorm["talker.model.layers.1.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_0 --> P_talker_model_layers_1_input_layernorm
        P_talker_model_layers_1_self_attn["talker.model.layers.1.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_1_input_layernorm --> P_talker_model_layers_1_self_attn
        P_talker_model_layers_1_post_attention_layernorm["talker.model.layers.1.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_1_self_attn --> P_talker_model_layers_1_post_attention_layernorm
        P_talker_model_layers_1_mlp["talker.model.layers.1.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_1_post_attention_layernorm --> P_talker_model_layers_1_mlp
        P_talker_model_layers_1["talker.model.layers.1\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_1_mlp --> P_talker_model_layers_1
        P_talker_model_layers_2_input_layernorm["talker.model.layers.2.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_1 --> P_talker_model_layers_2_input_layernorm
        P_talker_model_layers_2_self_attn["talker.model.layers.2.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_2_input_layernorm --> P_talker_model_layers_2_self_attn
        P_talker_model_layers_2_post_attention_layernorm["talker.model.layers.2.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_2_self_attn --> P_talker_model_layers_2_post_attention_layernorm
        P_talker_model_layers_2_mlp["talker.model.layers.2.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_2_post_attention_layernorm --> P_talker_model_layers_2_mlp
        P_talker_model_layers_2["talker.model.layers.2\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_2_mlp --> P_talker_model_layers_2
        P_talker_model_layers_3_input_layernorm["talker.model.layers.3.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_2 --> P_talker_model_layers_3_input_layernorm
        P_talker_model_layers_3_self_attn["talker.model.layers.3.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_3_input_layernorm --> P_talker_model_layers_3_self_attn
        P_talker_model_layers_3_post_attention_layernorm["talker.model.layers.3.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_3_self_attn --> P_talker_model_layers_3_post_attention_layernorm
        P_talker_model_layers_3_mlp["talker.model.layers.3.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_3_post_attention_layernorm --> P_talker_model_layers_3_mlp
        P_talker_model_layers_3["talker.model.layers.3\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_3_mlp --> P_talker_model_layers_3
        P_talker_model_layers_4_input_layernorm["talker.model.layers.4.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_3 --> P_talker_model_layers_4_input_layernorm
        P_talker_model_layers_4_self_attn["talker.model.layers.4.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_4_input_layernorm --> P_talker_model_layers_4_self_attn
        P_talker_model_layers_4_post_attention_layernorm["talker.model.layers.4.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_4_self_attn --> P_talker_model_layers_4_post_attention_layernorm
        P_talker_model_layers_4_mlp["talker.model.layers.4.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_4_post_attention_layernorm --> P_talker_model_layers_4_mlp
        P_talker_model_layers_4["talker.model.layers.4\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_4_mlp --> P_talker_model_layers_4
        P_talker_model_layers_5_input_layernorm["talker.model.layers.5.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_4 --> P_talker_model_layers_5_input_layernorm
        P_talker_model_layers_5_self_attn["talker.model.layers.5.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_5_input_layernorm --> P_talker_model_layers_5_self_attn
        P_talker_model_layers_5_post_attention_layernorm["talker.model.layers.5.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_5_self_attn --> P_talker_model_layers_5_post_attention_layernorm
        P_talker_model_layers_5_mlp["talker.model.layers.5.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_5_post_attention_layernorm --> P_talker_model_layers_5_mlp
        P_talker_model_layers_5["talker.model.layers.5\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_5_mlp --> P_talker_model_layers_5
        P_talker_model_layers_6_input_layernorm["talker.model.layers.6.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_5 --> P_talker_model_layers_6_input_layernorm
        P_talker_model_layers_6_self_attn["talker.model.layers.6.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_6_input_layernorm --> P_talker_model_layers_6_self_attn
        P_talker_model_layers_6_post_attention_layernorm["talker.model.layers.6.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_6_self_attn --> P_talker_model_layers_6_post_attention_layernorm
        P_talker_model_layers_6_mlp["talker.model.layers.6.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_6_post_attention_layernorm --> P_talker_model_layers_6_mlp
        P_talker_model_layers_6["talker.model.layers.6\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_6_mlp --> P_talker_model_layers_6
        P_talker_model_layers_7_input_layernorm["talker.model.layers.7.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_6 --> P_talker_model_layers_7_input_layernorm
        P_talker_model_layers_7_self_attn["talker.model.layers.7.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_7_input_layernorm --> P_talker_model_layers_7_self_attn
        P_talker_model_layers_7_post_attention_layernorm["talker.model.layers.7.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_7_self_attn --> P_talker_model_layers_7_post_attention_layernorm
        P_talker_model_layers_7_mlp["talker.model.layers.7.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_7_post_attention_layernorm --> P_talker_model_layers_7_mlp
        P_talker_model_layers_7["talker.model.layers.7\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_7_mlp --> P_talker_model_layers_7
        P_talker_model_layers_8_input_layernorm["talker.model.layers.8.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_7 --> P_talker_model_layers_8_input_layernorm
        P_talker_model_layers_8_self_attn["talker.model.layers.8.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_8_input_layernorm --> P_talker_model_layers_8_self_attn
        P_talker_model_layers_8_post_attention_layernorm["talker.model.layers.8.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_8_self_attn --> P_talker_model_layers_8_post_attention_layernorm
        P_talker_model_layers_8_mlp["talker.model.layers.8.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_8_post_attention_layernorm --> P_talker_model_layers_8_mlp
        P_talker_model_layers_8["talker.model.layers.8\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_8_mlp --> P_talker_model_layers_8
        P_talker_model_layers_9_input_layernorm["talker.model.layers.9.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_8 --> P_talker_model_layers_9_input_layernorm
        P_talker_model_layers_9_self_attn["talker.model.layers.9.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_9_input_layernorm --> P_talker_model_layers_9_self_attn
        P_talker_model_layers_9_post_attention_layernorm["talker.model.layers.9.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_9_self_attn --> P_talker_model_layers_9_post_attention_layernorm
        P_talker_model_layers_9_mlp["talker.model.layers.9.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_9_post_attention_layernorm --> P_talker_model_layers_9_mlp
        P_talker_model_layers_9["talker.model.layers.9\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_9_mlp --> P_talker_model_layers_9
        P_talker_model_layers_10_input_layernorm["talker.model.layers.10.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_9 --> P_talker_model_layers_10_input_layernorm
        P_talker_model_layers_10_self_attn["talker.model.layers.10.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_10_input_layernorm --> P_talker_model_layers_10_self_attn
        P_talker_model_layers_10_post_attention_layernorm["talker.model.layers.10.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_10_self_attn --> P_talker_model_layers_10_post_attention_layernorm
        P_talker_model_layers_10_mlp["talker.model.layers.10.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_10_post_attention_layernorm --> P_talker_model_layers_10_mlp
        P_talker_model_layers_10["talker.model.layers.10\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_10_mlp --> P_talker_model_layers_10
        P_talker_model_layers_11_input_layernorm["talker.model.layers.11.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_10 --> P_talker_model_layers_11_input_layernorm
        P_talker_model_layers_11_self_attn["talker.model.layers.11.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_11_input_layernorm --> P_talker_model_layers_11_self_attn
        P_talker_model_layers_11_post_attention_layernorm["talker.model.layers.11.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_11_self_attn --> P_talker_model_layers_11_post_attention_layernorm
        P_talker_model_layers_11_mlp["talker.model.layers.11.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_11_post_attention_layernorm --> P_talker_model_layers_11_mlp
        P_talker_model_layers_11["talker.model.layers.11\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_11_mlp --> P_talker_model_layers_11
        P_talker_model_layers_12_input_layernorm["talker.model.layers.12.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_11 --> P_talker_model_layers_12_input_layernorm
        P_talker_model_layers_12_self_attn["talker.model.layers.12.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_12_input_layernorm --> P_talker_model_layers_12_self_attn
        P_talker_model_layers_12_post_attention_layernorm["talker.model.layers.12.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_12_self_attn --> P_talker_model_layers_12_post_attention_layernorm
        P_talker_model_layers_12_mlp["talker.model.layers.12.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_12_post_attention_layernorm --> P_talker_model_layers_12_mlp
        P_talker_model_layers_12["talker.model.layers.12\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_12_mlp --> P_talker_model_layers_12
        P_talker_model_layers_13_input_layernorm["talker.model.layers.13.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_12 --> P_talker_model_layers_13_input_layernorm
        P_talker_model_layers_13_self_attn["talker.model.layers.13.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_13_input_layernorm --> P_talker_model_layers_13_self_attn
        P_talker_model_layers_13_post_attention_layernorm["talker.model.layers.13.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_13_self_attn --> P_talker_model_layers_13_post_attention_layernorm
        P_talker_model_layers_13_mlp["talker.model.layers.13.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_13_post_attention_layernorm --> P_talker_model_layers_13_mlp
        P_talker_model_layers_13["talker.model.layers.13\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_13_mlp --> P_talker_model_layers_13
        P_talker_model_layers_14_input_layernorm["talker.model.layers.14.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_13 --> P_talker_model_layers_14_input_layernorm
        P_talker_model_layers_14_self_attn["talker.model.layers.14.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_14_input_layernorm --> P_talker_model_layers_14_self_attn
        P_talker_model_layers_14_post_attention_layernorm["talker.model.layers.14.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_14_self_attn --> P_talker_model_layers_14_post_attention_layernorm
        P_talker_model_layers_14_mlp["talker.model.layers.14.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_14_post_attention_layernorm --> P_talker_model_layers_14_mlp
        P_talker_model_layers_14["talker.model.layers.14\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_14_mlp --> P_talker_model_layers_14
        P_talker_model_layers_15_input_layernorm["talker.model.layers.15.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_14 --> P_talker_model_layers_15_input_layernorm
        P_talker_model_layers_15_self_attn["talker.model.layers.15.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_15_input_layernorm --> P_talker_model_layers_15_self_attn
        P_talker_model_layers_15_post_attention_layernorm["talker.model.layers.15.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_15_self_attn --> P_talker_model_layers_15_post_attention_layernorm
        P_talker_model_layers_15_mlp["talker.model.layers.15.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_15_post_attention_layernorm --> P_talker_model_layers_15_mlp
        P_talker_model_layers_15["talker.model.layers.15\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_15_mlp --> P_talker_model_layers_15
        P_talker_model_layers_16_input_layernorm["talker.model.layers.16.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_15 --> P_talker_model_layers_16_input_layernorm
        P_talker_model_layers_16_self_attn["talker.model.layers.16.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_16_input_layernorm --> P_talker_model_layers_16_self_attn
        P_talker_model_layers_16_post_attention_layernorm["talker.model.layers.16.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_16_self_attn --> P_talker_model_layers_16_post_attention_layernorm
        P_talker_model_layers_16_mlp["talker.model.layers.16.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_16_post_attention_layernorm --> P_talker_model_layers_16_mlp
        P_talker_model_layers_16["talker.model.layers.16\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_16_mlp --> P_talker_model_layers_16
        P_talker_model_layers_17_input_layernorm["talker.model.layers.17.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_16 --> P_talker_model_layers_17_input_layernorm
        P_talker_model_layers_17_self_attn["talker.model.layers.17.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_17_input_layernorm --> P_talker_model_layers_17_self_attn
        P_talker_model_layers_17_post_attention_layernorm["talker.model.layers.17.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_17_self_attn --> P_talker_model_layers_17_post_attention_layernorm
        P_talker_model_layers_17_mlp["talker.model.layers.17.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_17_post_attention_layernorm --> P_talker_model_layers_17_mlp
        P_talker_model_layers_17["talker.model.layers.17\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_17_mlp --> P_talker_model_layers_17
        P_talker_model_layers_18_input_layernorm["talker.model.layers.18.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_17 --> P_talker_model_layers_18_input_layernorm
        P_talker_model_layers_18_self_attn["talker.model.layers.18.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_18_input_layernorm --> P_talker_model_layers_18_self_attn
        P_talker_model_layers_18_post_attention_layernorm["talker.model.layers.18.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_18_self_attn --> P_talker_model_layers_18_post_attention_layernorm
        P_talker_model_layers_18_mlp["talker.model.layers.18.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_18_post_attention_layernorm --> P_talker_model_layers_18_mlp
        P_talker_model_layers_18["talker.model.layers.18\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_18_mlp --> P_talker_model_layers_18
        P_talker_model_layers_19_input_layernorm["talker.model.layers.19.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_18 --> P_talker_model_layers_19_input_layernorm
        P_talker_model_layers_19_self_attn["talker.model.layers.19.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_talker_model_layers_19_input_layernorm --> P_talker_model_layers_19_self_attn
        P_talker_model_layers_19_post_attention_layernorm["talker.model.layers.19.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_19_self_attn --> P_talker_model_layers_19_post_attention_layernorm
        P_talker_model_layers_19_mlp["talker.model.layers.19.mlp\nType: Qwen3OmniMoeTalkerTextSparseMoeBlock\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_19_post_attention_layernorm --> P_talker_model_layers_19_mlp
        P_talker_model_layers_19["talker.model.layers.19\nType: Qwen3OmniMoeTalkerDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_19_mlp --> P_talker_model_layers_19
        P_talker_model_norm["talker.model.norm\nType: Qwen3OmniMoeTextRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_model_layers_19 --> P_talker_model_norm
        P_talker_model["talker.model\nQwen3OmniMoeTalkerModel"]
        P_talker_model_norm --> P_talker_model
        P_talker_codec_head["talker.codec_head\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x3072"]
        P_talker_model --> P_talker_codec_head
        P_talker["talker\nQwen3OmniMoeTalkerForConditionalGeneration"]
        P_talker_codec_head --> P_talker
        P_talker_model_codec_embedding["talker.model.codec_embedding\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker --> P_talker_model_codec_embedding
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x2x1024 bfloat16\nOut: 1x2x128"]
        P_talker_model_codec_embedding --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x2x1024 bfloat16\nOut: 1x2x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_0["talker.code_predictor.lm_head.0\nType: Linear\nIn: 1x2x1024 bfloat16\nOut: 1x2x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_0
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_0 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_0["talker.code_predictor.model.codec_embedding.0\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_0
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_0 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_1["talker.code_predictor.lm_head.1\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_1
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_1 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_1["talker.code_predictor.model.codec_embedding.1\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_1
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_1 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_2["talker.code_predictor.lm_head.2\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_2
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_2 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_2["talker.code_predictor.model.codec_embedding.2\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_2
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_2 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_3["talker.code_predictor.lm_head.3\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_3
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_3 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_3["talker.code_predictor.model.codec_embedding.3\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_3
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_3 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_4["talker.code_predictor.lm_head.4\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_4
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_4 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_4["talker.code_predictor.model.codec_embedding.4\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_4
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_4 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_5["talker.code_predictor.lm_head.5\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_5
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_5 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_5["talker.code_predictor.model.codec_embedding.5\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_5
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_5 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_6["talker.code_predictor.lm_head.6\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_6
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_6 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_6["talker.code_predictor.model.codec_embedding.6\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_6
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_6 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_7["talker.code_predictor.lm_head.7\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_7
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_7 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_7["talker.code_predictor.model.codec_embedding.7\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_7
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_7 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_8["talker.code_predictor.lm_head.8\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_8
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_8 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_8["talker.code_predictor.model.codec_embedding.8\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_8
        P_talker_code_predictor_model_rotary_emb["talker.code_predictor.model.rotary_emb\nType: Qwen3OmniMoeRotaryEmbedding\nIn: 1x1x1024 bfloat16\nOut: 1x1x128"]
        P_talker_code_predictor_model_codec_embedding_8 --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_layers_0["talker.code_predictor.model.layers.0\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_layers_0
        P_talker_code_predictor_model_layers_1["talker.code_predictor.model.layers.1\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_0 --> P_talker_code_predictor_model_layers_1
        P_talker_code_predictor_model_layers_2["talker.code_predictor.model.layers.2\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_1 --> P_talker_code_predictor_model_layers_2
        P_talker_code_predictor_model_layers_3["talker.code_predictor.model.layers.3\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_2 --> P_talker_code_predictor_model_layers_3
        P_talker_code_predictor_model_layers_4["talker.code_predictor.model.layers.4\nType: Qwen3OmniMoeTalkerCodePredictorDecoderLayer\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_3 --> P_talker_code_predictor_model_layers_4
        P_talker_code_predictor_model_norm["talker.code_predictor.model.norm\nType: Qwen3OmniMoeRMSNorm\nIn: 1x1x1024 bfloat16\nOut: 1x1x1024"]
        P_talker_code_predictor_model_layers_4 --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nQwen3OmniMoeTalkerCodePredictorModel"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_9["talker.code_predictor.lm_head.9\nType: Linear\nIn: 1x1x1024 bfloat16\nOut: 1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_9
        P_talker_code_predictor["talker.code_predictor\nQwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"]
        P_talker_code_predictor_lm_head_9 --> P_talker_code_predictor
        P_talker_code_predictor_model_codec_embedding_9["talker.code_predictor.model.codec_embedding.9\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x1024"]
        P_talker_code_predictor --> P_talker_code_predictor_model_codec_embedding_9
        P_truncated["... truncated (max 499 edges)"]
        P_talker_code_predictor_model_codec_embedding_9 --> P_truncated
    end
```
