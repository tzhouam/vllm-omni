# Qwen3 Omni Model Structure - Thinker

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nType: Embedding\nIn: 1x14 int64\nOut: 1x14x2048"]
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nType: Qwen3OmniMoeThinkerTextRotaryEmbedding\nIn: 1x14x2048 bfloat16\nOut: 1x14x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0_input_layernorm["thinker.model.layers.0.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0_input_layernorm
        P_thinker_model_layers_0_self_attn["thinker.model.layers.0.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_0_input_layernorm --> P_thinker_model_layers_0_self_attn
        P_thinker_model_layers_0_post_attention_layernorm["thinker.model.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_0_self_attn --> P_thinker_model_layers_0_post_attention_layernorm
        P_thinker_model_layers_0_mlp["thinker.model.layers.0.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_0_post_attention_layernorm --> P_thinker_model_layers_0_mlp
        P_thinker_model_layers_0["thinker.model.layers.0\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_0_mlp --> P_thinker_model_layers_0
        P_thinker_model_layers_1_input_layernorm["thinker.model.layers.1.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1_input_layernorm
        P_thinker_model_layers_1_self_attn["thinker.model.layers.1.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_1_input_layernorm --> P_thinker_model_layers_1_self_attn
        P_thinker_model_layers_1_post_attention_layernorm["thinker.model.layers.1.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_1_self_attn --> P_thinker_model_layers_1_post_attention_layernorm
        P_thinker_model_layers_1_mlp["thinker.model.layers.1.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_1_post_attention_layernorm --> P_thinker_model_layers_1_mlp
        P_thinker_model_layers_1["thinker.model.layers.1\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_1_mlp --> P_thinker_model_layers_1
        P_thinker_model_layers_2_input_layernorm["thinker.model.layers.2.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2_input_layernorm
        P_thinker_model_layers_2_self_attn["thinker.model.layers.2.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_2_input_layernorm --> P_thinker_model_layers_2_self_attn
        P_thinker_model_layers_2_post_attention_layernorm["thinker.model.layers.2.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_2_self_attn --> P_thinker_model_layers_2_post_attention_layernorm
        P_thinker_model_layers_2_mlp["thinker.model.layers.2.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_2_post_attention_layernorm --> P_thinker_model_layers_2_mlp
        P_thinker_model_layers_2["thinker.model.layers.2\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_2_mlp --> P_thinker_model_layers_2
        P_thinker_model_layers_3_input_layernorm["thinker.model.layers.3.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3_input_layernorm
        P_thinker_model_layers_3_self_attn["thinker.model.layers.3.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_3_input_layernorm --> P_thinker_model_layers_3_self_attn
        P_thinker_model_layers_3_post_attention_layernorm["thinker.model.layers.3.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_3_self_attn --> P_thinker_model_layers_3_post_attention_layernorm
        P_thinker_model_layers_3_mlp["thinker.model.layers.3.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_3_post_attention_layernorm --> P_thinker_model_layers_3_mlp
        P_thinker_model_layers_3["thinker.model.layers.3\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_3_mlp --> P_thinker_model_layers_3
        P_thinker_model_layers_4_input_layernorm["thinker.model.layers.4.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4_input_layernorm
        P_thinker_model_layers_4_self_attn["thinker.model.layers.4.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_4_input_layernorm --> P_thinker_model_layers_4_self_attn
        P_thinker_model_layers_4_post_attention_layernorm["thinker.model.layers.4.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_4_self_attn --> P_thinker_model_layers_4_post_attention_layernorm
        P_thinker_model_layers_4_mlp["thinker.model.layers.4.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_4_post_attention_layernorm --> P_thinker_model_layers_4_mlp
        P_thinker_model_layers_4["thinker.model.layers.4\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_4_mlp --> P_thinker_model_layers_4
        P_thinker_model_layers_5_input_layernorm["thinker.model.layers.5.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5_input_layernorm
        P_thinker_model_layers_5_self_attn["thinker.model.layers.5.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_5_input_layernorm --> P_thinker_model_layers_5_self_attn
        P_thinker_model_layers_5_post_attention_layernorm["thinker.model.layers.5.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_5_self_attn --> P_thinker_model_layers_5_post_attention_layernorm
        P_thinker_model_layers_5_mlp["thinker.model.layers.5.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_5_post_attention_layernorm --> P_thinker_model_layers_5_mlp
        P_thinker_model_layers_5["thinker.model.layers.5\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_5_mlp --> P_thinker_model_layers_5
        P_thinker_model_layers_6_input_layernorm["thinker.model.layers.6.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6_input_layernorm
        P_thinker_model_layers_6_self_attn["thinker.model.layers.6.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_6_input_layernorm --> P_thinker_model_layers_6_self_attn
        P_thinker_model_layers_6_post_attention_layernorm["thinker.model.layers.6.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_6_self_attn --> P_thinker_model_layers_6_post_attention_layernorm
        P_thinker_model_layers_6_mlp["thinker.model.layers.6.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_6_post_attention_layernorm --> P_thinker_model_layers_6_mlp
        P_thinker_model_layers_6["thinker.model.layers.6\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_6_mlp --> P_thinker_model_layers_6
        P_thinker_model_layers_7_input_layernorm["thinker.model.layers.7.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7_input_layernorm
        P_thinker_model_layers_7_self_attn["thinker.model.layers.7.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_7_input_layernorm --> P_thinker_model_layers_7_self_attn
        P_thinker_model_layers_7_post_attention_layernorm["thinker.model.layers.7.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_7_self_attn --> P_thinker_model_layers_7_post_attention_layernorm
        P_thinker_model_layers_7_mlp["thinker.model.layers.7.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_7_post_attention_layernorm --> P_thinker_model_layers_7_mlp
        P_thinker_model_layers_7["thinker.model.layers.7\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_7_mlp --> P_thinker_model_layers_7
        P_thinker_model_layers_8_input_layernorm["thinker.model.layers.8.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8_input_layernorm
        P_thinker_model_layers_8_self_attn["thinker.model.layers.8.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_8_input_layernorm --> P_thinker_model_layers_8_self_attn
        P_thinker_model_layers_8_post_attention_layernorm["thinker.model.layers.8.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_8_self_attn --> P_thinker_model_layers_8_post_attention_layernorm
        P_thinker_model_layers_8_mlp["thinker.model.layers.8.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_8_post_attention_layernorm --> P_thinker_model_layers_8_mlp
        P_thinker_model_layers_8["thinker.model.layers.8\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_8_mlp --> P_thinker_model_layers_8
        P_thinker_model_layers_9_input_layernorm["thinker.model.layers.9.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9_input_layernorm
        P_thinker_model_layers_9_self_attn["thinker.model.layers.9.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_9_input_layernorm --> P_thinker_model_layers_9_self_attn
        P_thinker_model_layers_9_post_attention_layernorm["thinker.model.layers.9.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_9_self_attn --> P_thinker_model_layers_9_post_attention_layernorm
        P_thinker_model_layers_9_mlp["thinker.model.layers.9.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_9_post_attention_layernorm --> P_thinker_model_layers_9_mlp
        P_thinker_model_layers_9["thinker.model.layers.9\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_9_mlp --> P_thinker_model_layers_9
        P_thinker_model_layers_10_input_layernorm["thinker.model.layers.10.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10_input_layernorm
        P_thinker_model_layers_10_self_attn["thinker.model.layers.10.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_10_input_layernorm --> P_thinker_model_layers_10_self_attn
        P_thinker_model_layers_10_post_attention_layernorm["thinker.model.layers.10.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_10_self_attn --> P_thinker_model_layers_10_post_attention_layernorm
        P_thinker_model_layers_10_mlp["thinker.model.layers.10.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_10_post_attention_layernorm --> P_thinker_model_layers_10_mlp
        P_thinker_model_layers_10["thinker.model.layers.10\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_10_mlp --> P_thinker_model_layers_10
        P_thinker_model_layers_11_input_layernorm["thinker.model.layers.11.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11_input_layernorm
        P_thinker_model_layers_11_self_attn["thinker.model.layers.11.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_11_input_layernorm --> P_thinker_model_layers_11_self_attn
        P_thinker_model_layers_11_post_attention_layernorm["thinker.model.layers.11.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_11_self_attn --> P_thinker_model_layers_11_post_attention_layernorm
        P_thinker_model_layers_11_mlp["thinker.model.layers.11.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_11_post_attention_layernorm --> P_thinker_model_layers_11_mlp
        P_thinker_model_layers_11["thinker.model.layers.11\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_11_mlp --> P_thinker_model_layers_11
        P_thinker_model_layers_12_input_layernorm["thinker.model.layers.12.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12_input_layernorm
        P_thinker_model_layers_12_self_attn["thinker.model.layers.12.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_12_input_layernorm --> P_thinker_model_layers_12_self_attn
        P_thinker_model_layers_12_post_attention_layernorm["thinker.model.layers.12.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_12_self_attn --> P_thinker_model_layers_12_post_attention_layernorm
        P_thinker_model_layers_12_mlp["thinker.model.layers.12.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_12_post_attention_layernorm --> P_thinker_model_layers_12_mlp
        P_thinker_model_layers_12["thinker.model.layers.12\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_12_mlp --> P_thinker_model_layers_12
        P_thinker_model_layers_13_input_layernorm["thinker.model.layers.13.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13_input_layernorm
        P_thinker_model_layers_13_self_attn["thinker.model.layers.13.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_13_input_layernorm --> P_thinker_model_layers_13_self_attn
        P_thinker_model_layers_13_post_attention_layernorm["thinker.model.layers.13.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_13_self_attn --> P_thinker_model_layers_13_post_attention_layernorm
        P_thinker_model_layers_13_mlp["thinker.model.layers.13.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_13_post_attention_layernorm --> P_thinker_model_layers_13_mlp
        P_thinker_model_layers_13["thinker.model.layers.13\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_13_mlp --> P_thinker_model_layers_13
        P_thinker_model_layers_14_input_layernorm["thinker.model.layers.14.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14_input_layernorm
        P_thinker_model_layers_14_self_attn["thinker.model.layers.14.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_14_input_layernorm --> P_thinker_model_layers_14_self_attn
        P_thinker_model_layers_14_post_attention_layernorm["thinker.model.layers.14.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_14_self_attn --> P_thinker_model_layers_14_post_attention_layernorm
        P_thinker_model_layers_14_mlp["thinker.model.layers.14.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_14_post_attention_layernorm --> P_thinker_model_layers_14_mlp
        P_thinker_model_layers_14["thinker.model.layers.14\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_14_mlp --> P_thinker_model_layers_14
        P_thinker_model_layers_15_input_layernorm["thinker.model.layers.15.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15_input_layernorm
        P_thinker_model_layers_15_self_attn["thinker.model.layers.15.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_15_input_layernorm --> P_thinker_model_layers_15_self_attn
        P_thinker_model_layers_15_post_attention_layernorm["thinker.model.layers.15.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_15_self_attn --> P_thinker_model_layers_15_post_attention_layernorm
        P_thinker_model_layers_15_mlp["thinker.model.layers.15.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_15_post_attention_layernorm --> P_thinker_model_layers_15_mlp
        P_thinker_model_layers_15["thinker.model.layers.15\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_15_mlp --> P_thinker_model_layers_15
        P_thinker_model_layers_16_input_layernorm["thinker.model.layers.16.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16_input_layernorm
        P_thinker_model_layers_16_self_attn["thinker.model.layers.16.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_16_input_layernorm --> P_thinker_model_layers_16_self_attn
        P_thinker_model_layers_16_post_attention_layernorm["thinker.model.layers.16.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_16_self_attn --> P_thinker_model_layers_16_post_attention_layernorm
        P_thinker_model_layers_16_mlp["thinker.model.layers.16.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_16_post_attention_layernorm --> P_thinker_model_layers_16_mlp
        P_thinker_model_layers_16["thinker.model.layers.16\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_16_mlp --> P_thinker_model_layers_16
        P_thinker_model_layers_17_input_layernorm["thinker.model.layers.17.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17_input_layernorm
        P_thinker_model_layers_17_self_attn["thinker.model.layers.17.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_17_input_layernorm --> P_thinker_model_layers_17_self_attn
        P_thinker_model_layers_17_post_attention_layernorm["thinker.model.layers.17.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_17_self_attn --> P_thinker_model_layers_17_post_attention_layernorm
        P_thinker_model_layers_17_mlp["thinker.model.layers.17.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_17_post_attention_layernorm --> P_thinker_model_layers_17_mlp
        P_thinker_model_layers_17["thinker.model.layers.17\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_17_mlp --> P_thinker_model_layers_17
        P_thinker_model_layers_18_input_layernorm["thinker.model.layers.18.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18_input_layernorm
        P_thinker_model_layers_18_self_attn["thinker.model.layers.18.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_18_input_layernorm --> P_thinker_model_layers_18_self_attn
        P_thinker_model_layers_18_post_attention_layernorm["thinker.model.layers.18.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_18_self_attn --> P_thinker_model_layers_18_post_attention_layernorm
        P_thinker_model_layers_18_mlp["thinker.model.layers.18.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_18_post_attention_layernorm --> P_thinker_model_layers_18_mlp
        P_thinker_model_layers_18["thinker.model.layers.18\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_18_mlp --> P_thinker_model_layers_18
        P_thinker_model_layers_19_input_layernorm["thinker.model.layers.19.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19_input_layernorm
        P_thinker_model_layers_19_self_attn["thinker.model.layers.19.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_19_input_layernorm --> P_thinker_model_layers_19_self_attn
        P_thinker_model_layers_19_post_attention_layernorm["thinker.model.layers.19.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_19_self_attn --> P_thinker_model_layers_19_post_attention_layernorm
        P_thinker_model_layers_19_mlp["thinker.model.layers.19.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_19_post_attention_layernorm --> P_thinker_model_layers_19_mlp
        P_thinker_model_layers_19["thinker.model.layers.19\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_19_mlp --> P_thinker_model_layers_19
        P_thinker_model_layers_20_input_layernorm["thinker.model.layers.20.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20_input_layernorm
        P_thinker_model_layers_20_self_attn["thinker.model.layers.20.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_20_input_layernorm --> P_thinker_model_layers_20_self_attn
        P_thinker_model_layers_20_post_attention_layernorm["thinker.model.layers.20.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_20_self_attn --> P_thinker_model_layers_20_post_attention_layernorm
        P_thinker_model_layers_20_mlp["thinker.model.layers.20.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_20_post_attention_layernorm --> P_thinker_model_layers_20_mlp
        P_thinker_model_layers_20["thinker.model.layers.20\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_20_mlp --> P_thinker_model_layers_20
        P_thinker_model_layers_21_input_layernorm["thinker.model.layers.21.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21_input_layernorm
        P_thinker_model_layers_21_self_attn["thinker.model.layers.21.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_21_input_layernorm --> P_thinker_model_layers_21_self_attn
        P_thinker_model_layers_21_post_attention_layernorm["thinker.model.layers.21.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_21_self_attn --> P_thinker_model_layers_21_post_attention_layernorm
        P_thinker_model_layers_21_mlp["thinker.model.layers.21.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_21_post_attention_layernorm --> P_thinker_model_layers_21_mlp
        P_thinker_model_layers_21["thinker.model.layers.21\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_21_mlp --> P_thinker_model_layers_21
        P_thinker_model_layers_22_input_layernorm["thinker.model.layers.22.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22_input_layernorm
        P_thinker_model_layers_22_self_attn["thinker.model.layers.22.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_22_input_layernorm --> P_thinker_model_layers_22_self_attn
        P_thinker_model_layers_22_post_attention_layernorm["thinker.model.layers.22.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_22_self_attn --> P_thinker_model_layers_22_post_attention_layernorm
        P_thinker_model_layers_22_mlp["thinker.model.layers.22.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_22_post_attention_layernorm --> P_thinker_model_layers_22_mlp
        P_thinker_model_layers_22["thinker.model.layers.22\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_22_mlp --> P_thinker_model_layers_22
        P_thinker_model_layers_23_input_layernorm["thinker.model.layers.23.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23_input_layernorm
        P_thinker_model_layers_23_self_attn["thinker.model.layers.23.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_23_input_layernorm --> P_thinker_model_layers_23_self_attn
        P_thinker_model_layers_23_post_attention_layernorm["thinker.model.layers.23.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_23_self_attn --> P_thinker_model_layers_23_post_attention_layernorm
        P_thinker_model_layers_23_mlp["thinker.model.layers.23.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_23_post_attention_layernorm --> P_thinker_model_layers_23_mlp
        P_thinker_model_layers_23["thinker.model.layers.23\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_23_mlp --> P_thinker_model_layers_23
        P_thinker_model_layers_24_input_layernorm["thinker.model.layers.24.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24_input_layernorm
        P_thinker_model_layers_24_self_attn["thinker.model.layers.24.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_24_input_layernorm --> P_thinker_model_layers_24_self_attn
        P_thinker_model_layers_24_post_attention_layernorm["thinker.model.layers.24.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_24_self_attn --> P_thinker_model_layers_24_post_attention_layernorm
        P_thinker_model_layers_24_mlp["thinker.model.layers.24.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_24_post_attention_layernorm --> P_thinker_model_layers_24_mlp
        P_thinker_model_layers_24["thinker.model.layers.24\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_24_mlp --> P_thinker_model_layers_24
        P_thinker_model_layers_25_input_layernorm["thinker.model.layers.25.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25_input_layernorm
        P_thinker_model_layers_25_self_attn["thinker.model.layers.25.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_25_input_layernorm --> P_thinker_model_layers_25_self_attn
        P_thinker_model_layers_25_post_attention_layernorm["thinker.model.layers.25.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_25_self_attn --> P_thinker_model_layers_25_post_attention_layernorm
        P_thinker_model_layers_25_mlp["thinker.model.layers.25.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_25_post_attention_layernorm --> P_thinker_model_layers_25_mlp
        P_thinker_model_layers_25["thinker.model.layers.25\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_25_mlp --> P_thinker_model_layers_25
        P_thinker_model_layers_26_input_layernorm["thinker.model.layers.26.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26_input_layernorm
        P_thinker_model_layers_26_self_attn["thinker.model.layers.26.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_26_input_layernorm --> P_thinker_model_layers_26_self_attn
        P_thinker_model_layers_26_post_attention_layernorm["thinker.model.layers.26.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_26_self_attn --> P_thinker_model_layers_26_post_attention_layernorm
        P_thinker_model_layers_26_mlp["thinker.model.layers.26.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_26_post_attention_layernorm --> P_thinker_model_layers_26_mlp
        P_thinker_model_layers_26["thinker.model.layers.26\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_26_mlp --> P_thinker_model_layers_26
        P_thinker_model_layers_27_input_layernorm["thinker.model.layers.27.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27_input_layernorm
        P_thinker_model_layers_27_self_attn["thinker.model.layers.27.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_27_input_layernorm --> P_thinker_model_layers_27_self_attn
        P_thinker_model_layers_27_post_attention_layernorm["thinker.model.layers.27.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_27_self_attn --> P_thinker_model_layers_27_post_attention_layernorm
        P_thinker_model_layers_27_mlp["thinker.model.layers.27.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_27_post_attention_layernorm --> P_thinker_model_layers_27_mlp
        P_thinker_model_layers_27["thinker.model.layers.27\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_27_mlp --> P_thinker_model_layers_27
        P_thinker_model_layers_28_input_layernorm["thinker.model.layers.28.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28_input_layernorm
        P_thinker_model_layers_28_self_attn["thinker.model.layers.28.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_28_input_layernorm --> P_thinker_model_layers_28_self_attn
        P_thinker_model_layers_28_post_attention_layernorm["thinker.model.layers.28.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_28_self_attn --> P_thinker_model_layers_28_post_attention_layernorm
        P_thinker_model_layers_28_mlp["thinker.model.layers.28.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_28_post_attention_layernorm --> P_thinker_model_layers_28_mlp
        P_thinker_model_layers_28["thinker.model.layers.28\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_28_mlp --> P_thinker_model_layers_28
        P_thinker_model_layers_29_input_layernorm["thinker.model.layers.29.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29_input_layernorm
        P_thinker_model_layers_29_self_attn["thinker.model.layers.29.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_29_input_layernorm --> P_thinker_model_layers_29_self_attn
        P_thinker_model_layers_29_post_attention_layernorm["thinker.model.layers.29.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_29_self_attn --> P_thinker_model_layers_29_post_attention_layernorm
        P_thinker_model_layers_29_mlp["thinker.model.layers.29.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_29_post_attention_layernorm --> P_thinker_model_layers_29_mlp
        P_thinker_model_layers_29["thinker.model.layers.29\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_29_mlp --> P_thinker_model_layers_29
        P_thinker_model_layers_30_input_layernorm["thinker.model.layers.30.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30_input_layernorm
        P_thinker_model_layers_30_self_attn["thinker.model.layers.30.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_30_input_layernorm --> P_thinker_model_layers_30_self_attn
        P_thinker_model_layers_30_post_attention_layernorm["thinker.model.layers.30.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_30_self_attn --> P_thinker_model_layers_30_post_attention_layernorm
        P_thinker_model_layers_30_mlp["thinker.model.layers.30.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_30_post_attention_layernorm --> P_thinker_model_layers_30_mlp
        P_thinker_model_layers_30["thinker.model.layers.30\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_30_mlp --> P_thinker_model_layers_30
        P_thinker_model_layers_31_input_layernorm["thinker.model.layers.31.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31_input_layernorm
        P_thinker_model_layers_31_self_attn["thinker.model.layers.31.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_31_input_layernorm --> P_thinker_model_layers_31_self_attn
        P_thinker_model_layers_31_post_attention_layernorm["thinker.model.layers.31.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_31_self_attn --> P_thinker_model_layers_31_post_attention_layernorm
        P_thinker_model_layers_31_mlp["thinker.model.layers.31.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_31_post_attention_layernorm --> P_thinker_model_layers_31_mlp
        P_thinker_model_layers_31["thinker.model.layers.31\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_31_mlp --> P_thinker_model_layers_31
        P_thinker_model_layers_32_input_layernorm["thinker.model.layers.32.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32_input_layernorm
        P_thinker_model_layers_32_self_attn["thinker.model.layers.32.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_32_input_layernorm --> P_thinker_model_layers_32_self_attn
        P_thinker_model_layers_32_post_attention_layernorm["thinker.model.layers.32.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_32_self_attn --> P_thinker_model_layers_32_post_attention_layernorm
        P_thinker_model_layers_32_mlp["thinker.model.layers.32.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_32_post_attention_layernorm --> P_thinker_model_layers_32_mlp
        P_thinker_model_layers_32["thinker.model.layers.32\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_32_mlp --> P_thinker_model_layers_32
        P_thinker_model_layers_33_input_layernorm["thinker.model.layers.33.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33_input_layernorm
        P_thinker_model_layers_33_self_attn["thinker.model.layers.33.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_33_input_layernorm --> P_thinker_model_layers_33_self_attn
        P_thinker_model_layers_33_post_attention_layernorm["thinker.model.layers.33.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_33_self_attn --> P_thinker_model_layers_33_post_attention_layernorm
        P_thinker_model_layers_33_mlp["thinker.model.layers.33.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_33_post_attention_layernorm --> P_thinker_model_layers_33_mlp
        P_thinker_model_layers_33["thinker.model.layers.33\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_33_mlp --> P_thinker_model_layers_33
        P_thinker_model_layers_34_input_layernorm["thinker.model.layers.34.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34_input_layernorm
        P_thinker_model_layers_34_self_attn["thinker.model.layers.34.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_34_input_layernorm --> P_thinker_model_layers_34_self_attn
        P_thinker_model_layers_34_post_attention_layernorm["thinker.model.layers.34.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_34_self_attn --> P_thinker_model_layers_34_post_attention_layernorm
        P_thinker_model_layers_34_mlp["thinker.model.layers.34.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_34_post_attention_layernorm --> P_thinker_model_layers_34_mlp
        P_thinker_model_layers_34["thinker.model.layers.34\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_34_mlp --> P_thinker_model_layers_34
        P_thinker_model_layers_35_input_layernorm["thinker.model.layers.35.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35_input_layernorm
        P_thinker_model_layers_35_self_attn["thinker.model.layers.35.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_35_input_layernorm --> P_thinker_model_layers_35_self_attn
        P_thinker_model_layers_35_post_attention_layernorm["thinker.model.layers.35.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_35_self_attn --> P_thinker_model_layers_35_post_attention_layernorm
        P_thinker_model_layers_35_mlp["thinker.model.layers.35.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_35_post_attention_layernorm --> P_thinker_model_layers_35_mlp
        P_thinker_model_layers_35["thinker.model.layers.35\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_35_mlp --> P_thinker_model_layers_35
        P_thinker_model_layers_36_input_layernorm["thinker.model.layers.36.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_35 --> P_thinker_model_layers_36_input_layernorm
        P_thinker_model_layers_36_self_attn["thinker.model.layers.36.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_36_input_layernorm --> P_thinker_model_layers_36_self_attn
        P_thinker_model_layers_36_post_attention_layernorm["thinker.model.layers.36.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_36_self_attn --> P_thinker_model_layers_36_post_attention_layernorm
        P_thinker_model_layers_36_mlp["thinker.model.layers.36.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_36_post_attention_layernorm --> P_thinker_model_layers_36_mlp
        P_thinker_model_layers_36["thinker.model.layers.36\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_36_mlp --> P_thinker_model_layers_36
        P_thinker_model_layers_37_input_layernorm["thinker.model.layers.37.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_36 --> P_thinker_model_layers_37_input_layernorm
        P_thinker_model_layers_37_self_attn["thinker.model.layers.37.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_37_input_layernorm --> P_thinker_model_layers_37_self_attn
        P_thinker_model_layers_37_post_attention_layernorm["thinker.model.layers.37.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_37_self_attn --> P_thinker_model_layers_37_post_attention_layernorm
        P_thinker_model_layers_37_mlp["thinker.model.layers.37.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_37_post_attention_layernorm --> P_thinker_model_layers_37_mlp
        P_thinker_model_layers_37["thinker.model.layers.37\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_37_mlp --> P_thinker_model_layers_37
        P_thinker_model_layers_38_input_layernorm["thinker.model.layers.38.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_37 --> P_thinker_model_layers_38_input_layernorm
        P_thinker_model_layers_38_self_attn["thinker.model.layers.38.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_38_input_layernorm --> P_thinker_model_layers_38_self_attn
        P_thinker_model_layers_38_post_attention_layernorm["thinker.model.layers.38.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_38_self_attn --> P_thinker_model_layers_38_post_attention_layernorm
        P_thinker_model_layers_38_mlp["thinker.model.layers.38.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_38_post_attention_layernorm --> P_thinker_model_layers_38_mlp
        P_thinker_model_layers_38["thinker.model.layers.38\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_38_mlp --> P_thinker_model_layers_38
        P_thinker_model_layers_39_input_layernorm["thinker.model.layers.39.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_38 --> P_thinker_model_layers_39_input_layernorm
        P_thinker_model_layers_39_self_attn["thinker.model.layers.39.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_39_input_layernorm --> P_thinker_model_layers_39_self_attn
        P_thinker_model_layers_39_post_attention_layernorm["thinker.model.layers.39.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_39_self_attn --> P_thinker_model_layers_39_post_attention_layernorm
        P_thinker_model_layers_39_mlp["thinker.model.layers.39.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_39_post_attention_layernorm --> P_thinker_model_layers_39_mlp
        P_thinker_model_layers_39["thinker.model.layers.39\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_39_mlp --> P_thinker_model_layers_39
        P_thinker_model_layers_40_input_layernorm["thinker.model.layers.40.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_39 --> P_thinker_model_layers_40_input_layernorm
        P_thinker_model_layers_40_self_attn["thinker.model.layers.40.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_40_input_layernorm --> P_thinker_model_layers_40_self_attn
        P_thinker_model_layers_40_post_attention_layernorm["thinker.model.layers.40.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_40_self_attn --> P_thinker_model_layers_40_post_attention_layernorm
        P_thinker_model_layers_40_mlp["thinker.model.layers.40.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_40_post_attention_layernorm --> P_thinker_model_layers_40_mlp
        P_thinker_model_layers_40["thinker.model.layers.40\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_40_mlp --> P_thinker_model_layers_40
        P_thinker_model_layers_41_input_layernorm["thinker.model.layers.41.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_40 --> P_thinker_model_layers_41_input_layernorm
        P_thinker_model_layers_41_self_attn["thinker.model.layers.41.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_41_input_layernorm --> P_thinker_model_layers_41_self_attn
        P_thinker_model_layers_41_post_attention_layernorm["thinker.model.layers.41.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_41_self_attn --> P_thinker_model_layers_41_post_attention_layernorm
        P_thinker_model_layers_41_mlp["thinker.model.layers.41.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_41_post_attention_layernorm --> P_thinker_model_layers_41_mlp
        P_thinker_model_layers_41["thinker.model.layers.41\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_41_mlp --> P_thinker_model_layers_41
        P_thinker_model_layers_42_input_layernorm["thinker.model.layers.42.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_41 --> P_thinker_model_layers_42_input_layernorm
        P_thinker_model_layers_42_self_attn["thinker.model.layers.42.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_42_input_layernorm --> P_thinker_model_layers_42_self_attn
        P_thinker_model_layers_42_post_attention_layernorm["thinker.model.layers.42.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_42_self_attn --> P_thinker_model_layers_42_post_attention_layernorm
        P_thinker_model_layers_42_mlp["thinker.model.layers.42.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_42_post_attention_layernorm --> P_thinker_model_layers_42_mlp
        P_thinker_model_layers_42["thinker.model.layers.42\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_42_mlp --> P_thinker_model_layers_42
        P_thinker_model_layers_43_input_layernorm["thinker.model.layers.43.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_42 --> P_thinker_model_layers_43_input_layernorm
        P_thinker_model_layers_43_self_attn["thinker.model.layers.43.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_43_input_layernorm --> P_thinker_model_layers_43_self_attn
        P_thinker_model_layers_43_post_attention_layernorm["thinker.model.layers.43.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_43_self_attn --> P_thinker_model_layers_43_post_attention_layernorm
        P_thinker_model_layers_43_mlp["thinker.model.layers.43.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_43_post_attention_layernorm --> P_thinker_model_layers_43_mlp
        P_thinker_model_layers_43["thinker.model.layers.43\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_43_mlp --> P_thinker_model_layers_43
        P_thinker_model_layers_44_input_layernorm["thinker.model.layers.44.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_43 --> P_thinker_model_layers_44_input_layernorm
        P_thinker_model_layers_44_self_attn["thinker.model.layers.44.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_44_input_layernorm --> P_thinker_model_layers_44_self_attn
        P_thinker_model_layers_44_post_attention_layernorm["thinker.model.layers.44.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_44_self_attn --> P_thinker_model_layers_44_post_attention_layernorm
        P_thinker_model_layers_44_mlp["thinker.model.layers.44.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_44_post_attention_layernorm --> P_thinker_model_layers_44_mlp
        P_thinker_model_layers_44["thinker.model.layers.44\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_44_mlp --> P_thinker_model_layers_44
        P_thinker_model_layers_45_input_layernorm["thinker.model.layers.45.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_44 --> P_thinker_model_layers_45_input_layernorm
        P_thinker_model_layers_45_self_attn["thinker.model.layers.45.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_45_input_layernorm --> P_thinker_model_layers_45_self_attn
        P_thinker_model_layers_45_post_attention_layernorm["thinker.model.layers.45.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_45_self_attn --> P_thinker_model_layers_45_post_attention_layernorm
        P_thinker_model_layers_45_mlp["thinker.model.layers.45.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_45_post_attention_layernorm --> P_thinker_model_layers_45_mlp
        P_thinker_model_layers_45["thinker.model.layers.45\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_45_mlp --> P_thinker_model_layers_45
        P_thinker_model_layers_46_input_layernorm["thinker.model.layers.46.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_45 --> P_thinker_model_layers_46_input_layernorm
        P_thinker_model_layers_46_self_attn["thinker.model.layers.46.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_46_input_layernorm --> P_thinker_model_layers_46_self_attn
        P_thinker_model_layers_46_post_attention_layernorm["thinker.model.layers.46.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_46_self_attn --> P_thinker_model_layers_46_post_attention_layernorm
        P_thinker_model_layers_46_mlp["thinker.model.layers.46.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_46_post_attention_layernorm --> P_thinker_model_layers_46_mlp
        P_thinker_model_layers_46["thinker.model.layers.46\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_46_mlp --> P_thinker_model_layers_46
        P_thinker_model_layers_47_input_layernorm["thinker.model.layers.47.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_46 --> P_thinker_model_layers_47_input_layernorm
        P_thinker_model_layers_47_self_attn["thinker.model.layers.47.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_47_input_layernorm --> P_thinker_model_layers_47_self_attn
        P_thinker_model_layers_47_post_attention_layernorm["thinker.model.layers.47.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_47_self_attn --> P_thinker_model_layers_47_post_attention_layernorm
        P_thinker_model_layers_47_mlp["thinker.model.layers.47.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_47_post_attention_layernorm --> P_thinker_model_layers_47_mlp
        P_thinker_model_layers_47["thinker.model.layers.47\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_47_mlp --> P_thinker_model_layers_47
        P_thinker_model_norm["thinker.model.norm\nType: Qwen3OmniMoeTextRMSNorm\nIn: 1x14x2048 bfloat16\nOut: 1x14x2048"]
        P_thinker_model_layers_47 --> P_thinker_model_norm
        P_thinker_model["thinker.model\nQwen3OmniMoeThinkerTextModel"]
        P_thinker_model_norm --> P_thinker_model
        P_thinker_lm_head["thinker.lm_head\nType: Linear\nIn: 1x14x2048 bfloat16\nOut: 1x14x152064"]
        P_thinker_model --> P_thinker_lm_head
        P_thinker["thinker\nQwen3OmniMoeThinkerForConditionalGeneration"]
        P_thinker_lm_head --> P_thinker
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x2048"]
        P_thinker --> P_thinker_model_embed_tokens
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nType: Qwen3OmniMoeThinkerTextRotaryEmbedding\nIn: 1x1x2048 bfloat16\nOut: 1x1x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0_input_layernorm["thinker.model.layers.0.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0_input_layernorm
        P_thinker_model_layers_0_self_attn["thinker.model.layers.0.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_0_input_layernorm --> P_thinker_model_layers_0_self_attn
        P_thinker_model_layers_0_post_attention_layernorm["thinker.model.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_self_attn --> P_thinker_model_layers_0_post_attention_layernorm
        P_thinker_model_layers_0_mlp["thinker.model.layers.0.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_post_attention_layernorm --> P_thinker_model_layers_0_mlp
        P_thinker_model_layers_0["thinker.model.layers.0\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_mlp --> P_thinker_model_layers_0
        P_thinker_model_layers_1_input_layernorm["thinker.model.layers.1.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1_input_layernorm
        P_thinker_model_layers_1_self_attn["thinker.model.layers.1.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_1_input_layernorm --> P_thinker_model_layers_1_self_attn
        P_thinker_model_layers_1_post_attention_layernorm["thinker.model.layers.1.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_1_self_attn --> P_thinker_model_layers_1_post_attention_layernorm
        P_thinker_model_layers_1_mlp["thinker.model.layers.1.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_1_post_attention_layernorm --> P_thinker_model_layers_1_mlp
        P_thinker_model_layers_1["thinker.model.layers.1\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_1_mlp --> P_thinker_model_layers_1
        P_thinker_model_layers_2_input_layernorm["thinker.model.layers.2.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2_input_layernorm
        P_thinker_model_layers_2_self_attn["thinker.model.layers.2.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_2_input_layernorm --> P_thinker_model_layers_2_self_attn
        P_thinker_model_layers_2_post_attention_layernorm["thinker.model.layers.2.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_2_self_attn --> P_thinker_model_layers_2_post_attention_layernorm
        P_thinker_model_layers_2_mlp["thinker.model.layers.2.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_2_post_attention_layernorm --> P_thinker_model_layers_2_mlp
        P_thinker_model_layers_2["thinker.model.layers.2\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_2_mlp --> P_thinker_model_layers_2
        P_thinker_model_layers_3_input_layernorm["thinker.model.layers.3.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3_input_layernorm
        P_thinker_model_layers_3_self_attn["thinker.model.layers.3.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_3_input_layernorm --> P_thinker_model_layers_3_self_attn
        P_thinker_model_layers_3_post_attention_layernorm["thinker.model.layers.3.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_3_self_attn --> P_thinker_model_layers_3_post_attention_layernorm
        P_thinker_model_layers_3_mlp["thinker.model.layers.3.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_3_post_attention_layernorm --> P_thinker_model_layers_3_mlp
        P_thinker_model_layers_3["thinker.model.layers.3\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_3_mlp --> P_thinker_model_layers_3
        P_thinker_model_layers_4_input_layernorm["thinker.model.layers.4.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4_input_layernorm
        P_thinker_model_layers_4_self_attn["thinker.model.layers.4.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_4_input_layernorm --> P_thinker_model_layers_4_self_attn
        P_thinker_model_layers_4_post_attention_layernorm["thinker.model.layers.4.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_4_self_attn --> P_thinker_model_layers_4_post_attention_layernorm
        P_thinker_model_layers_4_mlp["thinker.model.layers.4.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_4_post_attention_layernorm --> P_thinker_model_layers_4_mlp
        P_thinker_model_layers_4["thinker.model.layers.4\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_4_mlp --> P_thinker_model_layers_4
        P_thinker_model_layers_5_input_layernorm["thinker.model.layers.5.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5_input_layernorm
        P_thinker_model_layers_5_self_attn["thinker.model.layers.5.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_5_input_layernorm --> P_thinker_model_layers_5_self_attn
        P_thinker_model_layers_5_post_attention_layernorm["thinker.model.layers.5.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_5_self_attn --> P_thinker_model_layers_5_post_attention_layernorm
        P_thinker_model_layers_5_mlp["thinker.model.layers.5.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_5_post_attention_layernorm --> P_thinker_model_layers_5_mlp
        P_thinker_model_layers_5["thinker.model.layers.5\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_5_mlp --> P_thinker_model_layers_5
        P_thinker_model_layers_6_input_layernorm["thinker.model.layers.6.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6_input_layernorm
        P_thinker_model_layers_6_self_attn["thinker.model.layers.6.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_6_input_layernorm --> P_thinker_model_layers_6_self_attn
        P_thinker_model_layers_6_post_attention_layernorm["thinker.model.layers.6.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_6_self_attn --> P_thinker_model_layers_6_post_attention_layernorm
        P_thinker_model_layers_6_mlp["thinker.model.layers.6.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_6_post_attention_layernorm --> P_thinker_model_layers_6_mlp
        P_thinker_model_layers_6["thinker.model.layers.6\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_6_mlp --> P_thinker_model_layers_6
        P_thinker_model_layers_7_input_layernorm["thinker.model.layers.7.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7_input_layernorm
        P_thinker_model_layers_7_self_attn["thinker.model.layers.7.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_7_input_layernorm --> P_thinker_model_layers_7_self_attn
        P_thinker_model_layers_7_post_attention_layernorm["thinker.model.layers.7.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_7_self_attn --> P_thinker_model_layers_7_post_attention_layernorm
        P_thinker_model_layers_7_mlp["thinker.model.layers.7.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_7_post_attention_layernorm --> P_thinker_model_layers_7_mlp
        P_thinker_model_layers_7["thinker.model.layers.7\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_7_mlp --> P_thinker_model_layers_7
        P_thinker_model_layers_8_input_layernorm["thinker.model.layers.8.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8_input_layernorm
        P_thinker_model_layers_8_self_attn["thinker.model.layers.8.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_8_input_layernorm --> P_thinker_model_layers_8_self_attn
        P_thinker_model_layers_8_post_attention_layernorm["thinker.model.layers.8.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_8_self_attn --> P_thinker_model_layers_8_post_attention_layernorm
        P_thinker_model_layers_8_mlp["thinker.model.layers.8.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_8_post_attention_layernorm --> P_thinker_model_layers_8_mlp
        P_thinker_model_layers_8["thinker.model.layers.8\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_8_mlp --> P_thinker_model_layers_8
        P_thinker_model_layers_9_input_layernorm["thinker.model.layers.9.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9_input_layernorm
        P_thinker_model_layers_9_self_attn["thinker.model.layers.9.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_9_input_layernorm --> P_thinker_model_layers_9_self_attn
        P_thinker_model_layers_9_post_attention_layernorm["thinker.model.layers.9.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_9_self_attn --> P_thinker_model_layers_9_post_attention_layernorm
        P_thinker_model_layers_9_mlp["thinker.model.layers.9.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_9_post_attention_layernorm --> P_thinker_model_layers_9_mlp
        P_thinker_model_layers_9["thinker.model.layers.9\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_9_mlp --> P_thinker_model_layers_9
        P_thinker_model_layers_10_input_layernorm["thinker.model.layers.10.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10_input_layernorm
        P_thinker_model_layers_10_self_attn["thinker.model.layers.10.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_10_input_layernorm --> P_thinker_model_layers_10_self_attn
        P_thinker_model_layers_10_post_attention_layernorm["thinker.model.layers.10.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_10_self_attn --> P_thinker_model_layers_10_post_attention_layernorm
        P_thinker_model_layers_10_mlp["thinker.model.layers.10.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_10_post_attention_layernorm --> P_thinker_model_layers_10_mlp
        P_thinker_model_layers_10["thinker.model.layers.10\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_10_mlp --> P_thinker_model_layers_10
        P_thinker_model_layers_11_input_layernorm["thinker.model.layers.11.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11_input_layernorm
        P_thinker_model_layers_11_self_attn["thinker.model.layers.11.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_11_input_layernorm --> P_thinker_model_layers_11_self_attn
        P_thinker_model_layers_11_post_attention_layernorm["thinker.model.layers.11.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_11_self_attn --> P_thinker_model_layers_11_post_attention_layernorm
        P_thinker_model_layers_11_mlp["thinker.model.layers.11.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_11_post_attention_layernorm --> P_thinker_model_layers_11_mlp
        P_thinker_model_layers_11["thinker.model.layers.11\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_11_mlp --> P_thinker_model_layers_11
        P_thinker_model_layers_12_input_layernorm["thinker.model.layers.12.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12_input_layernorm
        P_thinker_model_layers_12_self_attn["thinker.model.layers.12.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_12_input_layernorm --> P_thinker_model_layers_12_self_attn
        P_thinker_model_layers_12_post_attention_layernorm["thinker.model.layers.12.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_12_self_attn --> P_thinker_model_layers_12_post_attention_layernorm
        P_thinker_model_layers_12_mlp["thinker.model.layers.12.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_12_post_attention_layernorm --> P_thinker_model_layers_12_mlp
        P_thinker_model_layers_12["thinker.model.layers.12\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_12_mlp --> P_thinker_model_layers_12
        P_thinker_model_layers_13_input_layernorm["thinker.model.layers.13.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13_input_layernorm
        P_thinker_model_layers_13_self_attn["thinker.model.layers.13.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_13_input_layernorm --> P_thinker_model_layers_13_self_attn
        P_thinker_model_layers_13_post_attention_layernorm["thinker.model.layers.13.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_13_self_attn --> P_thinker_model_layers_13_post_attention_layernorm
        P_thinker_model_layers_13_mlp["thinker.model.layers.13.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_13_post_attention_layernorm --> P_thinker_model_layers_13_mlp
        P_thinker_model_layers_13["thinker.model.layers.13\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_13_mlp --> P_thinker_model_layers_13
        P_thinker_model_layers_14_input_layernorm["thinker.model.layers.14.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14_input_layernorm
        P_thinker_model_layers_14_self_attn["thinker.model.layers.14.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_14_input_layernorm --> P_thinker_model_layers_14_self_attn
        P_thinker_model_layers_14_post_attention_layernorm["thinker.model.layers.14.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_14_self_attn --> P_thinker_model_layers_14_post_attention_layernorm
        P_thinker_model_layers_14_mlp["thinker.model.layers.14.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_14_post_attention_layernorm --> P_thinker_model_layers_14_mlp
        P_thinker_model_layers_14["thinker.model.layers.14\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_14_mlp --> P_thinker_model_layers_14
        P_thinker_model_layers_15_input_layernorm["thinker.model.layers.15.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15_input_layernorm
        P_thinker_model_layers_15_self_attn["thinker.model.layers.15.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_15_input_layernorm --> P_thinker_model_layers_15_self_attn
        P_thinker_model_layers_15_post_attention_layernorm["thinker.model.layers.15.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_15_self_attn --> P_thinker_model_layers_15_post_attention_layernorm
        P_thinker_model_layers_15_mlp["thinker.model.layers.15.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_15_post_attention_layernorm --> P_thinker_model_layers_15_mlp
        P_thinker_model_layers_15["thinker.model.layers.15\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_15_mlp --> P_thinker_model_layers_15
        P_thinker_model_layers_16_input_layernorm["thinker.model.layers.16.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16_input_layernorm
        P_thinker_model_layers_16_self_attn["thinker.model.layers.16.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_16_input_layernorm --> P_thinker_model_layers_16_self_attn
        P_thinker_model_layers_16_post_attention_layernorm["thinker.model.layers.16.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_16_self_attn --> P_thinker_model_layers_16_post_attention_layernorm
        P_thinker_model_layers_16_mlp["thinker.model.layers.16.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_16_post_attention_layernorm --> P_thinker_model_layers_16_mlp
        P_thinker_model_layers_16["thinker.model.layers.16\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_16_mlp --> P_thinker_model_layers_16
        P_thinker_model_layers_17_input_layernorm["thinker.model.layers.17.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17_input_layernorm
        P_thinker_model_layers_17_self_attn["thinker.model.layers.17.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_17_input_layernorm --> P_thinker_model_layers_17_self_attn
        P_thinker_model_layers_17_post_attention_layernorm["thinker.model.layers.17.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_17_self_attn --> P_thinker_model_layers_17_post_attention_layernorm
        P_thinker_model_layers_17_mlp["thinker.model.layers.17.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_17_post_attention_layernorm --> P_thinker_model_layers_17_mlp
        P_thinker_model_layers_17["thinker.model.layers.17\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_17_mlp --> P_thinker_model_layers_17
        P_thinker_model_layers_18_input_layernorm["thinker.model.layers.18.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18_input_layernorm
        P_thinker_model_layers_18_self_attn["thinker.model.layers.18.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_18_input_layernorm --> P_thinker_model_layers_18_self_attn
        P_thinker_model_layers_18_post_attention_layernorm["thinker.model.layers.18.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_18_self_attn --> P_thinker_model_layers_18_post_attention_layernorm
        P_thinker_model_layers_18_mlp["thinker.model.layers.18.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_18_post_attention_layernorm --> P_thinker_model_layers_18_mlp
        P_thinker_model_layers_18["thinker.model.layers.18\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_18_mlp --> P_thinker_model_layers_18
        P_thinker_model_layers_19_input_layernorm["thinker.model.layers.19.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19_input_layernorm
        P_thinker_model_layers_19_self_attn["thinker.model.layers.19.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_19_input_layernorm --> P_thinker_model_layers_19_self_attn
        P_thinker_model_layers_19_post_attention_layernorm["thinker.model.layers.19.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_19_self_attn --> P_thinker_model_layers_19_post_attention_layernorm
        P_thinker_model_layers_19_mlp["thinker.model.layers.19.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_19_post_attention_layernorm --> P_thinker_model_layers_19_mlp
        P_thinker_model_layers_19["thinker.model.layers.19\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_19_mlp --> P_thinker_model_layers_19
        P_thinker_model_layers_20_input_layernorm["thinker.model.layers.20.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20_input_layernorm
        P_thinker_model_layers_20_self_attn["thinker.model.layers.20.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_20_input_layernorm --> P_thinker_model_layers_20_self_attn
        P_thinker_model_layers_20_post_attention_layernorm["thinker.model.layers.20.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_20_self_attn --> P_thinker_model_layers_20_post_attention_layernorm
        P_thinker_model_layers_20_mlp["thinker.model.layers.20.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_20_post_attention_layernorm --> P_thinker_model_layers_20_mlp
        P_thinker_model_layers_20["thinker.model.layers.20\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_20_mlp --> P_thinker_model_layers_20
        P_thinker_model_layers_21_input_layernorm["thinker.model.layers.21.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21_input_layernorm
        P_thinker_model_layers_21_self_attn["thinker.model.layers.21.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_21_input_layernorm --> P_thinker_model_layers_21_self_attn
        P_thinker_model_layers_21_post_attention_layernorm["thinker.model.layers.21.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_21_self_attn --> P_thinker_model_layers_21_post_attention_layernorm
        P_thinker_model_layers_21_mlp["thinker.model.layers.21.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_21_post_attention_layernorm --> P_thinker_model_layers_21_mlp
        P_thinker_model_layers_21["thinker.model.layers.21\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_21_mlp --> P_thinker_model_layers_21
        P_thinker_model_layers_22_input_layernorm["thinker.model.layers.22.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22_input_layernorm
        P_thinker_model_layers_22_self_attn["thinker.model.layers.22.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_22_input_layernorm --> P_thinker_model_layers_22_self_attn
        P_thinker_model_layers_22_post_attention_layernorm["thinker.model.layers.22.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_22_self_attn --> P_thinker_model_layers_22_post_attention_layernorm
        P_thinker_model_layers_22_mlp["thinker.model.layers.22.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_22_post_attention_layernorm --> P_thinker_model_layers_22_mlp
        P_thinker_model_layers_22["thinker.model.layers.22\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_22_mlp --> P_thinker_model_layers_22
        P_thinker_model_layers_23_input_layernorm["thinker.model.layers.23.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23_input_layernorm
        P_thinker_model_layers_23_self_attn["thinker.model.layers.23.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_23_input_layernorm --> P_thinker_model_layers_23_self_attn
        P_thinker_model_layers_23_post_attention_layernorm["thinker.model.layers.23.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_23_self_attn --> P_thinker_model_layers_23_post_attention_layernorm
        P_thinker_model_layers_23_mlp["thinker.model.layers.23.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_23_post_attention_layernorm --> P_thinker_model_layers_23_mlp
        P_thinker_model_layers_23["thinker.model.layers.23\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_23_mlp --> P_thinker_model_layers_23
        P_thinker_model_layers_24_input_layernorm["thinker.model.layers.24.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24_input_layernorm
        P_thinker_model_layers_24_self_attn["thinker.model.layers.24.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_24_input_layernorm --> P_thinker_model_layers_24_self_attn
        P_thinker_model_layers_24_post_attention_layernorm["thinker.model.layers.24.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_24_self_attn --> P_thinker_model_layers_24_post_attention_layernorm
        P_thinker_model_layers_24_mlp["thinker.model.layers.24.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_24_post_attention_layernorm --> P_thinker_model_layers_24_mlp
        P_thinker_model_layers_24["thinker.model.layers.24\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_24_mlp --> P_thinker_model_layers_24
        P_thinker_model_layers_25_input_layernorm["thinker.model.layers.25.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25_input_layernorm
        P_thinker_model_layers_25_self_attn["thinker.model.layers.25.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_25_input_layernorm --> P_thinker_model_layers_25_self_attn
        P_thinker_model_layers_25_post_attention_layernorm["thinker.model.layers.25.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_25_self_attn --> P_thinker_model_layers_25_post_attention_layernorm
        P_thinker_model_layers_25_mlp["thinker.model.layers.25.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_25_post_attention_layernorm --> P_thinker_model_layers_25_mlp
        P_thinker_model_layers_25["thinker.model.layers.25\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_25_mlp --> P_thinker_model_layers_25
        P_thinker_model_layers_26_input_layernorm["thinker.model.layers.26.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26_input_layernorm
        P_thinker_model_layers_26_self_attn["thinker.model.layers.26.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_26_input_layernorm --> P_thinker_model_layers_26_self_attn
        P_thinker_model_layers_26_post_attention_layernorm["thinker.model.layers.26.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_26_self_attn --> P_thinker_model_layers_26_post_attention_layernorm
        P_thinker_model_layers_26_mlp["thinker.model.layers.26.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_26_post_attention_layernorm --> P_thinker_model_layers_26_mlp
        P_thinker_model_layers_26["thinker.model.layers.26\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_26_mlp --> P_thinker_model_layers_26
        P_thinker_model_layers_27_input_layernorm["thinker.model.layers.27.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27_input_layernorm
        P_thinker_model_layers_27_self_attn["thinker.model.layers.27.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_27_input_layernorm --> P_thinker_model_layers_27_self_attn
        P_thinker_model_layers_27_post_attention_layernorm["thinker.model.layers.27.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_27_self_attn --> P_thinker_model_layers_27_post_attention_layernorm
        P_thinker_model_layers_27_mlp["thinker.model.layers.27.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_27_post_attention_layernorm --> P_thinker_model_layers_27_mlp
        P_thinker_model_layers_27["thinker.model.layers.27\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_27_mlp --> P_thinker_model_layers_27
        P_thinker_model_layers_28_input_layernorm["thinker.model.layers.28.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28_input_layernorm
        P_thinker_model_layers_28_self_attn["thinker.model.layers.28.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_28_input_layernorm --> P_thinker_model_layers_28_self_attn
        P_thinker_model_layers_28_post_attention_layernorm["thinker.model.layers.28.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_28_self_attn --> P_thinker_model_layers_28_post_attention_layernorm
        P_thinker_model_layers_28_mlp["thinker.model.layers.28.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_28_post_attention_layernorm --> P_thinker_model_layers_28_mlp
        P_thinker_model_layers_28["thinker.model.layers.28\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_28_mlp --> P_thinker_model_layers_28
        P_thinker_model_layers_29_input_layernorm["thinker.model.layers.29.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29_input_layernorm
        P_thinker_model_layers_29_self_attn["thinker.model.layers.29.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_29_input_layernorm --> P_thinker_model_layers_29_self_attn
        P_thinker_model_layers_29_post_attention_layernorm["thinker.model.layers.29.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_29_self_attn --> P_thinker_model_layers_29_post_attention_layernorm
        P_thinker_model_layers_29_mlp["thinker.model.layers.29.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_29_post_attention_layernorm --> P_thinker_model_layers_29_mlp
        P_thinker_model_layers_29["thinker.model.layers.29\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_29_mlp --> P_thinker_model_layers_29
        P_thinker_model_layers_30_input_layernorm["thinker.model.layers.30.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30_input_layernorm
        P_thinker_model_layers_30_self_attn["thinker.model.layers.30.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_30_input_layernorm --> P_thinker_model_layers_30_self_attn
        P_thinker_model_layers_30_post_attention_layernorm["thinker.model.layers.30.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_30_self_attn --> P_thinker_model_layers_30_post_attention_layernorm
        P_thinker_model_layers_30_mlp["thinker.model.layers.30.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_30_post_attention_layernorm --> P_thinker_model_layers_30_mlp
        P_thinker_model_layers_30["thinker.model.layers.30\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_30_mlp --> P_thinker_model_layers_30
        P_thinker_model_layers_31_input_layernorm["thinker.model.layers.31.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31_input_layernorm
        P_thinker_model_layers_31_self_attn["thinker.model.layers.31.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_31_input_layernorm --> P_thinker_model_layers_31_self_attn
        P_thinker_model_layers_31_post_attention_layernorm["thinker.model.layers.31.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_31_self_attn --> P_thinker_model_layers_31_post_attention_layernorm
        P_thinker_model_layers_31_mlp["thinker.model.layers.31.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_31_post_attention_layernorm --> P_thinker_model_layers_31_mlp
        P_thinker_model_layers_31["thinker.model.layers.31\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_31_mlp --> P_thinker_model_layers_31
        P_thinker_model_layers_32_input_layernorm["thinker.model.layers.32.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32_input_layernorm
        P_thinker_model_layers_32_self_attn["thinker.model.layers.32.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_32_input_layernorm --> P_thinker_model_layers_32_self_attn
        P_thinker_model_layers_32_post_attention_layernorm["thinker.model.layers.32.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_32_self_attn --> P_thinker_model_layers_32_post_attention_layernorm
        P_thinker_model_layers_32_mlp["thinker.model.layers.32.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_32_post_attention_layernorm --> P_thinker_model_layers_32_mlp
        P_thinker_model_layers_32["thinker.model.layers.32\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_32_mlp --> P_thinker_model_layers_32
        P_thinker_model_layers_33_input_layernorm["thinker.model.layers.33.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33_input_layernorm
        P_thinker_model_layers_33_self_attn["thinker.model.layers.33.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_33_input_layernorm --> P_thinker_model_layers_33_self_attn
        P_thinker_model_layers_33_post_attention_layernorm["thinker.model.layers.33.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_33_self_attn --> P_thinker_model_layers_33_post_attention_layernorm
        P_thinker_model_layers_33_mlp["thinker.model.layers.33.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_33_post_attention_layernorm --> P_thinker_model_layers_33_mlp
        P_thinker_model_layers_33["thinker.model.layers.33\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_33_mlp --> P_thinker_model_layers_33
        P_thinker_model_layers_34_input_layernorm["thinker.model.layers.34.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34_input_layernorm
        P_thinker_model_layers_34_self_attn["thinker.model.layers.34.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_34_input_layernorm --> P_thinker_model_layers_34_self_attn
        P_thinker_model_layers_34_post_attention_layernorm["thinker.model.layers.34.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_34_self_attn --> P_thinker_model_layers_34_post_attention_layernorm
        P_thinker_model_layers_34_mlp["thinker.model.layers.34.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_34_post_attention_layernorm --> P_thinker_model_layers_34_mlp
        P_thinker_model_layers_34["thinker.model.layers.34\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_34_mlp --> P_thinker_model_layers_34
        P_thinker_model_layers_35_input_layernorm["thinker.model.layers.35.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35_input_layernorm
        P_thinker_model_layers_35_self_attn["thinker.model.layers.35.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_35_input_layernorm --> P_thinker_model_layers_35_self_attn
        P_thinker_model_layers_35_post_attention_layernorm["thinker.model.layers.35.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_35_self_attn --> P_thinker_model_layers_35_post_attention_layernorm
        P_thinker_model_layers_35_mlp["thinker.model.layers.35.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_35_post_attention_layernorm --> P_thinker_model_layers_35_mlp
        P_thinker_model_layers_35["thinker.model.layers.35\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_35_mlp --> P_thinker_model_layers_35
        P_thinker_model_layers_36_input_layernorm["thinker.model.layers.36.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_35 --> P_thinker_model_layers_36_input_layernorm
        P_thinker_model_layers_36_self_attn["thinker.model.layers.36.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_36_input_layernorm --> P_thinker_model_layers_36_self_attn
        P_thinker_model_layers_36_post_attention_layernorm["thinker.model.layers.36.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_36_self_attn --> P_thinker_model_layers_36_post_attention_layernorm
        P_thinker_model_layers_36_mlp["thinker.model.layers.36.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_36_post_attention_layernorm --> P_thinker_model_layers_36_mlp
        P_thinker_model_layers_36["thinker.model.layers.36\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_36_mlp --> P_thinker_model_layers_36
        P_thinker_model_layers_37_input_layernorm["thinker.model.layers.37.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_36 --> P_thinker_model_layers_37_input_layernorm
        P_thinker_model_layers_37_self_attn["thinker.model.layers.37.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_37_input_layernorm --> P_thinker_model_layers_37_self_attn
        P_thinker_model_layers_37_post_attention_layernorm["thinker.model.layers.37.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_37_self_attn --> P_thinker_model_layers_37_post_attention_layernorm
        P_thinker_model_layers_37_mlp["thinker.model.layers.37.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_37_post_attention_layernorm --> P_thinker_model_layers_37_mlp
        P_thinker_model_layers_37["thinker.model.layers.37\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_37_mlp --> P_thinker_model_layers_37
        P_thinker_model_layers_38_input_layernorm["thinker.model.layers.38.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_37 --> P_thinker_model_layers_38_input_layernorm
        P_thinker_model_layers_38_self_attn["thinker.model.layers.38.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_38_input_layernorm --> P_thinker_model_layers_38_self_attn
        P_thinker_model_layers_38_post_attention_layernorm["thinker.model.layers.38.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_38_self_attn --> P_thinker_model_layers_38_post_attention_layernorm
        P_thinker_model_layers_38_mlp["thinker.model.layers.38.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_38_post_attention_layernorm --> P_thinker_model_layers_38_mlp
        P_thinker_model_layers_38["thinker.model.layers.38\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_38_mlp --> P_thinker_model_layers_38
        P_thinker_model_layers_39_input_layernorm["thinker.model.layers.39.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_38 --> P_thinker_model_layers_39_input_layernorm
        P_thinker_model_layers_39_self_attn["thinker.model.layers.39.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_39_input_layernorm --> P_thinker_model_layers_39_self_attn
        P_thinker_model_layers_39_post_attention_layernorm["thinker.model.layers.39.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_39_self_attn --> P_thinker_model_layers_39_post_attention_layernorm
        P_thinker_model_layers_39_mlp["thinker.model.layers.39.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_39_post_attention_layernorm --> P_thinker_model_layers_39_mlp
        P_thinker_model_layers_39["thinker.model.layers.39\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_39_mlp --> P_thinker_model_layers_39
        P_thinker_model_layers_40_input_layernorm["thinker.model.layers.40.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_39 --> P_thinker_model_layers_40_input_layernorm
        P_thinker_model_layers_40_self_attn["thinker.model.layers.40.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_40_input_layernorm --> P_thinker_model_layers_40_self_attn
        P_thinker_model_layers_40_post_attention_layernorm["thinker.model.layers.40.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_40_self_attn --> P_thinker_model_layers_40_post_attention_layernorm
        P_thinker_model_layers_40_mlp["thinker.model.layers.40.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_40_post_attention_layernorm --> P_thinker_model_layers_40_mlp
        P_thinker_model_layers_40["thinker.model.layers.40\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_40_mlp --> P_thinker_model_layers_40
        P_thinker_model_layers_41_input_layernorm["thinker.model.layers.41.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_40 --> P_thinker_model_layers_41_input_layernorm
        P_thinker_model_layers_41_self_attn["thinker.model.layers.41.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_41_input_layernorm --> P_thinker_model_layers_41_self_attn
        P_thinker_model_layers_41_post_attention_layernorm["thinker.model.layers.41.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_41_self_attn --> P_thinker_model_layers_41_post_attention_layernorm
        P_thinker_model_layers_41_mlp["thinker.model.layers.41.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_41_post_attention_layernorm --> P_thinker_model_layers_41_mlp
        P_thinker_model_layers_41["thinker.model.layers.41\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_41_mlp --> P_thinker_model_layers_41
        P_thinker_model_layers_42_input_layernorm["thinker.model.layers.42.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_41 --> P_thinker_model_layers_42_input_layernorm
        P_thinker_model_layers_42_self_attn["thinker.model.layers.42.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_42_input_layernorm --> P_thinker_model_layers_42_self_attn
        P_thinker_model_layers_42_post_attention_layernorm["thinker.model.layers.42.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_42_self_attn --> P_thinker_model_layers_42_post_attention_layernorm
        P_thinker_model_layers_42_mlp["thinker.model.layers.42.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_42_post_attention_layernorm --> P_thinker_model_layers_42_mlp
        P_thinker_model_layers_42["thinker.model.layers.42\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_42_mlp --> P_thinker_model_layers_42
        P_thinker_model_layers_43_input_layernorm["thinker.model.layers.43.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_42 --> P_thinker_model_layers_43_input_layernorm
        P_thinker_model_layers_43_self_attn["thinker.model.layers.43.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_43_input_layernorm --> P_thinker_model_layers_43_self_attn
        P_thinker_model_layers_43_post_attention_layernorm["thinker.model.layers.43.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_43_self_attn --> P_thinker_model_layers_43_post_attention_layernorm
        P_thinker_model_layers_43_mlp["thinker.model.layers.43.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_43_post_attention_layernorm --> P_thinker_model_layers_43_mlp
        P_thinker_model_layers_43["thinker.model.layers.43\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_43_mlp --> P_thinker_model_layers_43
        P_thinker_model_layers_44_input_layernorm["thinker.model.layers.44.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_43 --> P_thinker_model_layers_44_input_layernorm
        P_thinker_model_layers_44_self_attn["thinker.model.layers.44.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_44_input_layernorm --> P_thinker_model_layers_44_self_attn
        P_thinker_model_layers_44_post_attention_layernorm["thinker.model.layers.44.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_44_self_attn --> P_thinker_model_layers_44_post_attention_layernorm
        P_thinker_model_layers_44_mlp["thinker.model.layers.44.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_44_post_attention_layernorm --> P_thinker_model_layers_44_mlp
        P_thinker_model_layers_44["thinker.model.layers.44\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_44_mlp --> P_thinker_model_layers_44
        P_thinker_model_layers_45_input_layernorm["thinker.model.layers.45.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_44 --> P_thinker_model_layers_45_input_layernorm
        P_thinker_model_layers_45_self_attn["thinker.model.layers.45.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_45_input_layernorm --> P_thinker_model_layers_45_self_attn
        P_thinker_model_layers_45_post_attention_layernorm["thinker.model.layers.45.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_45_self_attn --> P_thinker_model_layers_45_post_attention_layernorm
        P_thinker_model_layers_45_mlp["thinker.model.layers.45.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_45_post_attention_layernorm --> P_thinker_model_layers_45_mlp
        P_thinker_model_layers_45["thinker.model.layers.45\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_45_mlp --> P_thinker_model_layers_45
        P_thinker_model_layers_46_input_layernorm["thinker.model.layers.46.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_45 --> P_thinker_model_layers_46_input_layernorm
        P_thinker_model_layers_46_self_attn["thinker.model.layers.46.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_46_input_layernorm --> P_thinker_model_layers_46_self_attn
        P_thinker_model_layers_46_post_attention_layernorm["thinker.model.layers.46.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_46_self_attn --> P_thinker_model_layers_46_post_attention_layernorm
        P_thinker_model_layers_46_mlp["thinker.model.layers.46.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_46_post_attention_layernorm --> P_thinker_model_layers_46_mlp
        P_thinker_model_layers_46["thinker.model.layers.46\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_46_mlp --> P_thinker_model_layers_46
        P_thinker_model_layers_47_input_layernorm["thinker.model.layers.47.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_46 --> P_thinker_model_layers_47_input_layernorm
        P_thinker_model_layers_47_self_attn["thinker.model.layers.47.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_47_input_layernorm --> P_thinker_model_layers_47_self_attn
        P_thinker_model_layers_47_post_attention_layernorm["thinker.model.layers.47.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_47_self_attn --> P_thinker_model_layers_47_post_attention_layernorm
        P_thinker_model_layers_47_mlp["thinker.model.layers.47.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_47_post_attention_layernorm --> P_thinker_model_layers_47_mlp
        P_thinker_model_layers_47["thinker.model.layers.47\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_47_mlp --> P_thinker_model_layers_47
        P_thinker_model_norm["thinker.model.norm\nType: Qwen3OmniMoeTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_47 --> P_thinker_model_norm
        P_thinker_model["thinker.model\nQwen3OmniMoeThinkerTextModel"]
        P_thinker_model_norm --> P_thinker_model
        P_thinker_lm_head["thinker.lm_head\nType: Linear\nIn: 1x1x2048 bfloat16\nOut: 1x1x152064"]
        P_thinker_model --> P_thinker_lm_head
        P_thinker["thinker\nQwen3OmniMoeThinkerForConditionalGeneration"]
        P_thinker_lm_head --> P_thinker
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nType: Embedding\nIn: 1x1 int64\nOut: 1x1x2048"]
        P_thinker --> P_thinker_model_embed_tokens
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nType: Qwen3OmniMoeThinkerTextRotaryEmbedding\nIn: 1x1x2048 bfloat16\nOut: 1x1x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0_input_layernorm["thinker.model.layers.0.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0_input_layernorm
        P_thinker_model_layers_0_self_attn["thinker.model.layers.0.self_attn\nQwen3OmniMoeThinkerTextAttention"]
        P_thinker_model_layers_0_input_layernorm --> P_thinker_model_layers_0_self_attn
        P_thinker_model_layers_0_post_attention_layernorm["thinker.model.layers.0.post_attention_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_self_attn --> P_thinker_model_layers_0_post_attention_layernorm
        P_thinker_model_layers_0_mlp["thinker.model.layers.0.mlp\nType: Qwen3OmniMoeThinkerTextSparseMoeBlock\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_post_attention_layernorm --> P_thinker_model_layers_0_mlp
        P_thinker_model_layers_0["thinker.model.layers.0\nType: Qwen3OmniMoeThinkerTextDecoderLayer\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0_mlp --> P_thinker_model_layers_0
        P_thinker_model_layers_1_input_layernorm["thinker.model.layers.1.input_layernorm\nType: Qwen3OmniMoeThinkerTextRMSNorm\nIn: 1x1x2048 bfloat16\nOut: 1x1x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1_input_layernorm
        P_truncated["... truncated (max 499 edges)"]
        P_thinker_model_layers_1_input_layernorm --> P_truncated
    end
```
