# Qwen3 Omni Model Structure - Talker

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nLinear\n1x3x2048->1x3x2048"]
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nSiLUActivation\n1x3x2048->1x3x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nLinear\n1x3x2048->1x3x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nTalkerResizeMLP\n1x3x2048->1x3x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nLinear\n11x2048->11x2048"]
        P_talker_text_projection --> P_talker_text_projection_linear_fc1
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nSiLUActivation\n11x2048->11x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nLinear\n11x2048->11x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nTalkerResizeMLP\n11x2048->11x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_text_projection_linear_fc1["talker.text_projection.linear_fc1\nLinear\n1x20x2048->1x20x2048"]
        P_talker_text_projection --> P_talker_text_projection_linear_fc1
        P_talker_text_projection_act_fn["talker.text_projection.act_fn\nSiLUActivation\n1x20x2048->1x20x2048"]
        P_talker_text_projection_linear_fc1 --> P_talker_text_projection_act_fn
        P_talker_text_projection_linear_fc2["talker.text_projection.linear_fc2\nLinear\n1x20x2048->1x20x1024"]
        P_talker_text_projection_act_fn --> P_talker_text_projection_linear_fc2
        P_talker_text_projection["talker.text_projection\nTalkerResizeMLP\n1x20x2048->1x20x1024"]
        P_talker_text_projection_linear_fc2 --> P_talker_text_projection
        P_talker_model_codec_embedding["talker.model.codec_embedding\nEmbedding\n1x6->1x6x1024"]
        P_talker_text_projection --> P_talker_model_codec_embedding
        P_talker_model_rotary_emb["talker.model.rotary_emb\nTalkerRotaryEmbeddin\n1x20x1024->1x20x128"]
        P_talker_model_codec_embedding --> P_talker_model_rotary_emb
        P_talker_model_layers_0["talker...layers.0\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_rotary_emb --> P_talker_model_layers_0
        P_talker_model_layers_1["talker...layers.1\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_0 --> P_talker_model_layers_1
        P_talker_model_layers_2["talker...layers.2\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_1 --> P_talker_model_layers_2
        P_talker_model_layers_3["talker...layers.3\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_2 --> P_talker_model_layers_3
        P_talker_model_layers_4["talker...layers.4\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_3 --> P_talker_model_layers_4
        P_talker_model_layers_5["talker...layers.5\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_4 --> P_talker_model_layers_5
        P_talker_model_layers_6["talker...layers.6\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_5 --> P_talker_model_layers_6
        P_talker_model_layers_7["talker...layers.7\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_6 --> P_talker_model_layers_7
        P_talker_model_layers_8["talker...layers.8\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_7 --> P_talker_model_layers_8
        P_talker_model_layers_9["talker...layers.9\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_8 --> P_talker_model_layers_9
        P_talker_model_layers_10["talker...layers.10\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_9 --> P_talker_model_layers_10
        P_talker_model_layers_11["talker...layers.11\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_10 --> P_talker_model_layers_11
        P_talker_model_layers_12["talker...layers.12\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_11 --> P_talker_model_layers_12
        P_talker_model_layers_13["talker...layers.13\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_12 --> P_talker_model_layers_13
        P_talker_model_layers_14["talker...layers.14\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_13 --> P_talker_model_layers_14
        P_talker_model_layers_15["talker...layers.15\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_14 --> P_talker_model_layers_15
        P_talker_model_layers_16["talker...layers.16\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_15 --> P_talker_model_layers_16
        P_talker_model_layers_17["talker...layers.17\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_16 --> P_talker_model_layers_17
        P_talker_model_layers_18["talker...layers.18\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_17 --> P_talker_model_layers_18
        P_talker_model_layers_19["talker...layers.19\nTalkerDecoderLayer\n1x20x1024->1x20x1024"]
        P_talker_model_layers_18 --> P_talker_model_layers_19
        P_talker_model_norm["talker.model.norm\nTextRMSNorm\n1x20x1024->1x20x1024"]
        P_talker_model_layers_19 --> P_talker_model_norm
        P_talker_model["talker.model\nTalkerModel"]
        P_talker_model_norm --> P_talker_model
        P_talker_codec_head["talker.codec_head\nLinear\n1x20x1024->1x20x3072"]
        P_talker_model --> P_talker_codec_head
        P_talker["talker\nTalkerForConditional"]
        P_talker_codec_head --> P_talker
        P_talker_model_codec_embedding["talker.model.codec_embedding\nEmbedding\n1x1->1x1x1024"]
        P_talker --> P_talker_model_codec_embedding
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x2x1024->1x2x128"]
        P_talker_model_codec_embedding --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x2x1024->1x2x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_0["talker...lm_head.0\nLinear\n1x2x1024->1x2x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_0
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_0 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_1["talker...lm_head.1\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_1
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_1 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_2["talker...lm_head.2\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_2
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_2 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_3["talker...lm_head.3\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_3
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_3 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_4["talker...lm_head.4\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_4
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_4 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_5["talker...lm_head.5\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_5
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_5 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_6["talker...lm_head.6\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_6
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_6 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_7["talker...lm_head.7\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_7
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_7 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_8["talker...lm_head.8\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_8
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_8 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_9["talker...lm_head.9\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_9
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_9 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_10["talker...lm_head.10\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_10
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_10 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_11["talker...lm_head.11\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_11
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_11 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_12["talker...lm_head.12\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_12
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_12 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_13["talker...lm_head.13\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_13
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_13 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_14["talker...lm_head.14\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_14
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_14 --> P_talker_code_predictor
        P_talker_model_rotary_emb["talker.model.rotary_emb\nTalkerRotaryEmbeddin\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_model_rotary_emb
        P_talker_model_layers_0["talker...layers.0\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_rotary_emb --> P_talker_model_layers_0
        P_talker_model_layers_1["talker...layers.1\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_0 --> P_talker_model_layers_1
        P_talker_model_layers_2["talker...layers.2\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_1 --> P_talker_model_layers_2
        P_talker_model_layers_3["talker...layers.3\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_2 --> P_talker_model_layers_3
        P_talker_model_layers_4["talker...layers.4\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_3 --> P_talker_model_layers_4
        P_talker_model_layers_5["talker...layers.5\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_4 --> P_talker_model_layers_5
        P_talker_model_layers_6["talker...layers.6\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_5 --> P_talker_model_layers_6
        P_talker_model_layers_7["talker...layers.7\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_6 --> P_talker_model_layers_7
        P_talker_model_layers_8["talker...layers.8\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_7 --> P_talker_model_layers_8
        P_talker_model_layers_9["talker...layers.9\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_8 --> P_talker_model_layers_9
        P_talker_model_layers_10["talker...layers.10\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_9 --> P_talker_model_layers_10
        P_talker_model_layers_11["talker...layers.11\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_10 --> P_talker_model_layers_11
        P_talker_model_layers_12["talker...layers.12\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_11 --> P_talker_model_layers_12
        P_talker_model_layers_13["talker...layers.13\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_12 --> P_talker_model_layers_13
        P_talker_model_layers_14["talker...layers.14\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_13 --> P_talker_model_layers_14
        P_talker_model_layers_15["talker...layers.15\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_14 --> P_talker_model_layers_15
        P_talker_model_layers_16["talker...layers.16\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_15 --> P_talker_model_layers_16
        P_talker_model_layers_17["talker...layers.17\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_16 --> P_talker_model_layers_17
        P_talker_model_layers_18["talker...layers.18\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_17 --> P_talker_model_layers_18
        P_talker_model_layers_19["talker...layers.19\nTalkerDecoderLayer\n1x1x1024->1x1x1024"]
        P_talker_model_layers_18 --> P_talker_model_layers_19
        P_talker_model_norm["talker.model.norm\nTextRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_model_layers_19 --> P_talker_model_norm
        P_talker_model["talker.model\nTalkerModel"]
        P_talker_model_norm --> P_talker_model
        P_talker_codec_head["talker.codec_head\nLinear\n1x1x1024->1x1x3072"]
        P_talker_model --> P_talker_codec_head
        P_talker["talker\nTalkerForConditional"]
        P_talker_codec_head --> P_talker
        P_talker_model_codec_embedding["talker.model.codec_embedding\nEmbedding\n1x1->1x1x1024"]
        P_talker --> P_talker_model_codec_embedding
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x2x1024->1x2x128"]
        P_talker_model_codec_embedding --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x2x1024->1x2x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_0["talker...lm_head.0\nLinear\n1x2x1024->1x2x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_0
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_0 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_1["talker...lm_head.1\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_1
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_1 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_2["talker...lm_head.2\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_2
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_2 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_3["talker...lm_head.3\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_3
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_3 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_4["talker...lm_head.4\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_4
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_4 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_5["talker...lm_head.5\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_5
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_5 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_6["talker...lm_head.6\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_6
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_6 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_7["talker...lm_head.7\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_7
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_7 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_8["talker...lm_head.8\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_8
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_8 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_9["talker...lm_head.9\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_9
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_9 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_10["talker...lm_head.10\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_10
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_10 --> P_talker_code_predictor
        P_talker_code_predictor_model_rotary_emb["talker...model.rotary_emb\nRotaryEmbedding\n1x1x1024->1x1x128"]
        P_talker_code_predictor --> P_talker_code_predictor_model_rotary_emb
        P_talker_code_predictor_model_norm["talker...model.norm\nRMSNorm\n1x1x1024->1x1x1024"]
        P_talker_code_predictor_model_rotary_emb --> P_talker_code_predictor_model_norm
        P_talker_code_predictor_model["talker.code_predictor.model\nTalkerCodePredictorM"]
        P_talker_code_predictor_model_norm --> P_talker_code_predictor_model
        P_talker_code_predictor_lm_head_11["talker...lm_head.11\nLinear\n1x1x1024->1x1x2048"]
        P_talker_code_predictor_model --> P_talker_code_predictor_lm_head_11
        P_talker_code_predictor["talker.code_predictor\nTalkerCodePredictorM"]
        P_talker_code_predictor_lm_head_11 --> P_talker_code_predictor
        P_truncated["... truncated (max 200 nodes / 499 edges)"]
        P_talker_code_predictor --> P_truncated
    end
```
