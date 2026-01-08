# Qwen3 Omni Model Structure - Thinker

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nEmbedding\n1x14->1x14x2048"]
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nThinkerTextRotaryEmb\n1x14x2048->1x14x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0["thinker...layers.0\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0
        P_thinker_model_layers_1["thinker...layers.1\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1
        P_thinker_model_layers_2["thinker...layers.2\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2
        P_thinker_model_layers_3["thinker...layers.3\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3
        P_thinker_model_layers_4["thinker...layers.4\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4
        P_thinker_model_layers_5["thinker...layers.5\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5
        P_thinker_model_layers_6["thinker...layers.6\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6
        P_thinker_model_layers_7["thinker...layers.7\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7
        P_thinker_model_layers_8["thinker...layers.8\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8
        P_thinker_model_layers_9["thinker...layers.9\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9
        P_thinker_model_layers_10["thinker...layers.10\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10
        P_thinker_model_layers_11["thinker...layers.11\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11
        P_thinker_model_layers_12["thinker...layers.12\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12
        P_thinker_model_layers_13["thinker...layers.13\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13
        P_thinker_model_layers_14["thinker...layers.14\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14
        P_thinker_model_layers_15["thinker...layers.15\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15
        P_thinker_model_layers_16["thinker...layers.16\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16
        P_thinker_model_layers_17["thinker...layers.17\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17
        P_thinker_model_layers_18["thinker...layers.18\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18
        P_thinker_model_layers_19["thinker...layers.19\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19
        P_thinker_model_layers_20["thinker...layers.20\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20
        P_thinker_model_layers_21["thinker...layers.21\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21
        P_thinker_model_layers_22["thinker...layers.22\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22
        P_thinker_model_layers_23["thinker...layers.23\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23
        P_thinker_model_layers_24["thinker...layers.24\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24
        P_thinker_model_layers_25["thinker...layers.25\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25
        P_thinker_model_layers_26["thinker...layers.26\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26
        P_thinker_model_layers_27["thinker...layers.27\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27
        P_thinker_model_layers_28["thinker...layers.28\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28
        P_thinker_model_layers_29["thinker...layers.29\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29
        P_thinker_model_layers_30["thinker...layers.30\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30
        P_thinker_model_layers_31["thinker...layers.31\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31
        P_thinker_model_layers_32["thinker...layers.32\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32
        P_thinker_model_layers_33["thinker...layers.33\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33
        P_thinker_model_layers_34["thinker...layers.34\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34
        P_thinker_model_layers_35["thinker...layers.35\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35
        P_thinker_model_layers_36["thinker...layers.36\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_35 --> P_thinker_model_layers_36
        P_thinker_model_layers_37["thinker...layers.37\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_36 --> P_thinker_model_layers_37
        P_thinker_model_layers_38["thinker...layers.38\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_37 --> P_thinker_model_layers_38
        P_thinker_model_layers_39["thinker...layers.39\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_38 --> P_thinker_model_layers_39
        P_thinker_model_layers_40["thinker...layers.40\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_39 --> P_thinker_model_layers_40
        P_thinker_model_layers_41["thinker...layers.41\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_40 --> P_thinker_model_layers_41
        P_thinker_model_layers_42["thinker...layers.42\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_41 --> P_thinker_model_layers_42
        P_thinker_model_layers_43["thinker...layers.43\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_42 --> P_thinker_model_layers_43
        P_thinker_model_layers_44["thinker...layers.44\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_43 --> P_thinker_model_layers_44
        P_thinker_model_layers_45["thinker...layers.45\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_44 --> P_thinker_model_layers_45
        P_thinker_model_layers_46["thinker...layers.46\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_45 --> P_thinker_model_layers_46
        P_thinker_model_layers_47["thinker...layers.47\nThinkerTextDecoderLa\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_46 --> P_thinker_model_layers_47
        P_thinker_model_norm["thinker.model.norm\nTextRMSNorm\n1x14x2048->1x14x2048"]
        P_thinker_model_layers_47 --> P_thinker_model_norm
        P_thinker_model["thinker.model\nThinkerTextModel"]
        P_thinker_model_norm --> P_thinker_model
        P_thinker_lm_head["thinker.lm_head\nLinear\n1x14x2048->1x14x152064"]
        P_thinker_model --> P_thinker_lm_head
        P_thinker["thinker\nThinkerForConditiona"]
        P_thinker_lm_head --> P_thinker
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nEmbedding\n1x1->1x1x2048"]
        P_thinker --> P_thinker_model_embed_tokens
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nThinkerTextRotaryEmb\n1x1x2048->1x1x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0["thinker...layers.0\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0
        P_thinker_model_layers_1["thinker...layers.1\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1
        P_thinker_model_layers_2["thinker...layers.2\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2
        P_thinker_model_layers_3["thinker...layers.3\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3
        P_thinker_model_layers_4["thinker...layers.4\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4
        P_thinker_model_layers_5["thinker...layers.5\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5
        P_thinker_model_layers_6["thinker...layers.6\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6
        P_thinker_model_layers_7["thinker...layers.7\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7
        P_thinker_model_layers_8["thinker...layers.8\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8
        P_thinker_model_layers_9["thinker...layers.9\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9
        P_thinker_model_layers_10["thinker...layers.10\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10
        P_thinker_model_layers_11["thinker...layers.11\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11
        P_thinker_model_layers_12["thinker...layers.12\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12
        P_thinker_model_layers_13["thinker...layers.13\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13
        P_thinker_model_layers_14["thinker...layers.14\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14
        P_thinker_model_layers_15["thinker...layers.15\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15
        P_thinker_model_layers_16["thinker...layers.16\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16
        P_thinker_model_layers_17["thinker...layers.17\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17
        P_thinker_model_layers_18["thinker...layers.18\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18
        P_thinker_model_layers_19["thinker...layers.19\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19
        P_thinker_model_layers_20["thinker...layers.20\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20
        P_thinker_model_layers_21["thinker...layers.21\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21
        P_thinker_model_layers_22["thinker...layers.22\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22
        P_thinker_model_layers_23["thinker...layers.23\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23
        P_thinker_model_layers_24["thinker...layers.24\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24
        P_thinker_model_layers_25["thinker...layers.25\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25
        P_thinker_model_layers_26["thinker...layers.26\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26
        P_thinker_model_layers_27["thinker...layers.27\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27
        P_thinker_model_layers_28["thinker...layers.28\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28
        P_thinker_model_layers_29["thinker...layers.29\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29
        P_thinker_model_layers_30["thinker...layers.30\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30
        P_thinker_model_layers_31["thinker...layers.31\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31
        P_thinker_model_layers_32["thinker...layers.32\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32
        P_thinker_model_layers_33["thinker...layers.33\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33
        P_thinker_model_layers_34["thinker...layers.34\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34
        P_thinker_model_layers_35["thinker...layers.35\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35
        P_thinker_model_layers_36["thinker...layers.36\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_35 --> P_thinker_model_layers_36
        P_thinker_model_layers_37["thinker...layers.37\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_36 --> P_thinker_model_layers_37
        P_thinker_model_layers_38["thinker...layers.38\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_37 --> P_thinker_model_layers_38
        P_thinker_model_layers_39["thinker...layers.39\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_38 --> P_thinker_model_layers_39
        P_thinker_model_layers_40["thinker...layers.40\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_39 --> P_thinker_model_layers_40
        P_thinker_model_layers_41["thinker...layers.41\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_40 --> P_thinker_model_layers_41
        P_thinker_model_layers_42["thinker...layers.42\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_41 --> P_thinker_model_layers_42
        P_thinker_model_layers_43["thinker...layers.43\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_42 --> P_thinker_model_layers_43
        P_thinker_model_layers_44["thinker...layers.44\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_43 --> P_thinker_model_layers_44
        P_thinker_model_layers_45["thinker...layers.45\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_44 --> P_thinker_model_layers_45
        P_thinker_model_layers_46["thinker...layers.46\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_45 --> P_thinker_model_layers_46
        P_thinker_model_layers_47["thinker...layers.47\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_46 --> P_thinker_model_layers_47
        P_thinker_model_norm["thinker.model.norm\nTextRMSNorm\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_47 --> P_thinker_model_norm
        P_thinker_model["thinker.model\nThinkerTextModel"]
        P_thinker_model_norm --> P_thinker_model
        P_thinker_lm_head["thinker.lm_head\nLinear\n1x1x2048->1x1x152064"]
        P_thinker_model --> P_thinker_lm_head
        P_thinker["thinker\nThinkerForConditiona"]
        P_thinker_lm_head --> P_thinker
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nEmbedding\n1x1->1x1x2048"]
        P_thinker --> P_thinker_model_embed_tokens
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nThinkerTextRotaryEmb\n1x1x2048->1x1x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0["thinker...layers.0\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0
        P_thinker_model_layers_1["thinker...layers.1\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1
        P_thinker_model_layers_2["thinker...layers.2\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2
        P_thinker_model_layers_3["thinker...layers.3\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3
        P_thinker_model_layers_4["thinker...layers.4\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4
        P_thinker_model_layers_5["thinker...layers.5\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5
        P_thinker_model_layers_6["thinker...layers.6\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6
        P_thinker_model_layers_7["thinker...layers.7\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7
        P_thinker_model_layers_8["thinker...layers.8\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8
        P_thinker_model_layers_9["thinker...layers.9\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9
        P_thinker_model_layers_10["thinker...layers.10\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10
        P_thinker_model_layers_11["thinker...layers.11\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11
        P_thinker_model_layers_12["thinker...layers.12\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12
        P_thinker_model_layers_13["thinker...layers.13\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13
        P_thinker_model_layers_14["thinker...layers.14\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14
        P_thinker_model_layers_15["thinker...layers.15\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15
        P_thinker_model_layers_16["thinker...layers.16\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16
        P_thinker_model_layers_17["thinker...layers.17\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17
        P_thinker_model_layers_18["thinker...layers.18\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18
        P_thinker_model_layers_19["thinker...layers.19\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19
        P_thinker_model_layers_20["thinker...layers.20\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20
        P_thinker_model_layers_21["thinker...layers.21\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21
        P_thinker_model_layers_22["thinker...layers.22\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22
        P_thinker_model_layers_23["thinker...layers.23\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23
        P_thinker_model_layers_24["thinker...layers.24\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24
        P_thinker_model_layers_25["thinker...layers.25\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25
        P_thinker_model_layers_26["thinker...layers.26\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26
        P_thinker_model_layers_27["thinker...layers.27\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27
        P_thinker_model_layers_28["thinker...layers.28\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28
        P_thinker_model_layers_29["thinker...layers.29\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29
        P_thinker_model_layers_30["thinker...layers.30\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30
        P_thinker_model_layers_31["thinker...layers.31\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31
        P_thinker_model_layers_32["thinker...layers.32\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32
        P_thinker_model_layers_33["thinker...layers.33\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33
        P_thinker_model_layers_34["thinker...layers.34\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34
        P_thinker_model_layers_35["thinker...layers.35\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35
        P_thinker_model_layers_36["thinker...layers.36\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_35 --> P_thinker_model_layers_36
        P_thinker_model_layers_37["thinker...layers.37\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_36 --> P_thinker_model_layers_37
        P_thinker_model_layers_38["thinker...layers.38\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_37 --> P_thinker_model_layers_38
        P_thinker_model_layers_39["thinker...layers.39\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_38 --> P_thinker_model_layers_39
        P_thinker_model_layers_40["thinker...layers.40\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_39 --> P_thinker_model_layers_40
        P_thinker_model_layers_41["thinker...layers.41\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_40 --> P_thinker_model_layers_41
        P_thinker_model_layers_42["thinker...layers.42\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_41 --> P_thinker_model_layers_42
        P_thinker_model_layers_43["thinker...layers.43\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_42 --> P_thinker_model_layers_43
        P_thinker_model_layers_44["thinker...layers.44\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_43 --> P_thinker_model_layers_44
        P_thinker_model_layers_45["thinker...layers.45\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_44 --> P_thinker_model_layers_45
        P_thinker_model_layers_46["thinker...layers.46\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_45 --> P_thinker_model_layers_46
        P_thinker_model_layers_47["thinker...layers.47\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_46 --> P_thinker_model_layers_47
        P_thinker_model_norm["thinker.model.norm\nTextRMSNorm\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_47 --> P_thinker_model_norm
        P_thinker_model["thinker.model\nThinkerTextModel"]
        P_thinker_model_norm --> P_thinker_model
        P_thinker_lm_head["thinker.lm_head\nLinear\n1x1x2048->1x1x152064"]
        P_thinker_model --> P_thinker_lm_head
        P_thinker["thinker\nThinkerForConditiona"]
        P_thinker_lm_head --> P_thinker
        P_thinker_model_embed_tokens["thinker.model.embed_tokens\nEmbedding\n1x1->1x1x2048"]
        P_thinker --> P_thinker_model_embed_tokens
        P_thinker_model_rotary_emb["thinker.model.rotary_emb\nThinkerTextRotaryEmb\n1x1x2048->1x1x128"]
        P_thinker_model_embed_tokens --> P_thinker_model_rotary_emb
        P_thinker_model_layers_0["thinker...layers.0\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_rotary_emb --> P_thinker_model_layers_0
        P_thinker_model_layers_1["thinker...layers.1\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_0 --> P_thinker_model_layers_1
        P_thinker_model_layers_2["thinker...layers.2\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_1 --> P_thinker_model_layers_2
        P_thinker_model_layers_3["thinker...layers.3\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_2 --> P_thinker_model_layers_3
        P_thinker_model_layers_4["thinker...layers.4\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_3 --> P_thinker_model_layers_4
        P_thinker_model_layers_5["thinker...layers.5\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_4 --> P_thinker_model_layers_5
        P_thinker_model_layers_6["thinker...layers.6\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_5 --> P_thinker_model_layers_6
        P_thinker_model_layers_7["thinker...layers.7\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_6 --> P_thinker_model_layers_7
        P_thinker_model_layers_8["thinker...layers.8\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_7 --> P_thinker_model_layers_8
        P_thinker_model_layers_9["thinker...layers.9\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_8 --> P_thinker_model_layers_9
        P_thinker_model_layers_10["thinker...layers.10\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_9 --> P_thinker_model_layers_10
        P_thinker_model_layers_11["thinker...layers.11\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_10 --> P_thinker_model_layers_11
        P_thinker_model_layers_12["thinker...layers.12\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_11 --> P_thinker_model_layers_12
        P_thinker_model_layers_13["thinker...layers.13\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_12 --> P_thinker_model_layers_13
        P_thinker_model_layers_14["thinker...layers.14\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_13 --> P_thinker_model_layers_14
        P_thinker_model_layers_15["thinker...layers.15\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_14 --> P_thinker_model_layers_15
        P_thinker_model_layers_16["thinker...layers.16\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_15 --> P_thinker_model_layers_16
        P_thinker_model_layers_17["thinker...layers.17\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_16 --> P_thinker_model_layers_17
        P_thinker_model_layers_18["thinker...layers.18\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_17 --> P_thinker_model_layers_18
        P_thinker_model_layers_19["thinker...layers.19\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_18 --> P_thinker_model_layers_19
        P_thinker_model_layers_20["thinker...layers.20\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_19 --> P_thinker_model_layers_20
        P_thinker_model_layers_21["thinker...layers.21\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_20 --> P_thinker_model_layers_21
        P_thinker_model_layers_22["thinker...layers.22\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_21 --> P_thinker_model_layers_22
        P_thinker_model_layers_23["thinker...layers.23\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_22 --> P_thinker_model_layers_23
        P_thinker_model_layers_24["thinker...layers.24\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_23 --> P_thinker_model_layers_24
        P_thinker_model_layers_25["thinker...layers.25\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_24 --> P_thinker_model_layers_25
        P_thinker_model_layers_26["thinker...layers.26\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_25 --> P_thinker_model_layers_26
        P_thinker_model_layers_27["thinker...layers.27\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_26 --> P_thinker_model_layers_27
        P_thinker_model_layers_28["thinker...layers.28\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_27 --> P_thinker_model_layers_28
        P_thinker_model_layers_29["thinker...layers.29\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_28 --> P_thinker_model_layers_29
        P_thinker_model_layers_30["thinker...layers.30\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_29 --> P_thinker_model_layers_30
        P_thinker_model_layers_31["thinker...layers.31\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_30 --> P_thinker_model_layers_31
        P_thinker_model_layers_32["thinker...layers.32\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_31 --> P_thinker_model_layers_32
        P_thinker_model_layers_33["thinker...layers.33\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_32 --> P_thinker_model_layers_33
        P_thinker_model_layers_34["thinker...layers.34\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_33 --> P_thinker_model_layers_34
        P_thinker_model_layers_35["thinker...layers.35\nThinkerTextDecoderLa\n1x1x2048->1x1x2048"]
        P_thinker_model_layers_34 --> P_thinker_model_layers_35
        P_truncated["... truncated (max 200 nodes / 499 edges)"]
        P_thinker_model_layers_35 --> P_truncated
    end
```
