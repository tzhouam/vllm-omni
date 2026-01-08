# Qwen3 Omni Model Structure - Code2wav

```mermaid
flowchart TD
    subgraph PPhase [Prefill Phase]
        P_code2wav_code_embedding["code2wav.code_embedding\nEmbedding\n1x16x60->1x16x60x1024"]
        P_code2wav_pre_transformer_rotary_emb["code2wav.pre_transformer.rotary_emb\nRotaryEmbedding\n1x60x1024->1x60x64"]
        P_code2wav_code_embedding --> P_code2wav_pre_transformer_rotary_emb
        P_code2wav_pre_transformer_layers_0["code2wav...layers.0\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_rotary_emb --> P_code2wav_pre_transformer_layers_0
        P_code2wav_pre_transformer_layers_1["code2wav...layers.1\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_0 --> P_code2wav_pre_transformer_layers_1
        P_code2wav_pre_transformer_layers_2["code2wav...layers.2\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_1 --> P_code2wav_pre_transformer_layers_2
        P_code2wav_pre_transformer_layers_3["code2wav...layers.3\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_2 --> P_code2wav_pre_transformer_layers_3
        P_code2wav_pre_transformer_layers_4["code2wav...layers.4\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_3 --> P_code2wav_pre_transformer_layers_4
        P_code2wav_pre_transformer_layers_5["code2wav...layers.5\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_4 --> P_code2wav_pre_transformer_layers_5
        P_code2wav_pre_transformer_layers_6["code2wav...layers.6\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_5 --> P_code2wav_pre_transformer_layers_6
        P_code2wav_pre_transformer_layers_7["code2wav...layers.7\nCode2WavTransformerL\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_6 --> P_code2wav_pre_transformer_layers_7
        P_code2wav_pre_transformer_norm["code2wav.pre_transformer.norm\nRMSNorm\n1x60x1024->1x60x1024"]
        P_code2wav_pre_transformer_layers_7 --> P_code2wav_pre_transformer_norm
        P_code2wav_pre_transformer["code2wav.pre_transformer\nCode2WavTransformerM"]
        P_code2wav_pre_transformer_norm --> P_code2wav_pre_transformer
        P_code2wav_upsample_0_0["code2wav...0.0\nCausalTransConvNet\n1x1024x60->1x1024x120"]
        P_code2wav_pre_transformer --> P_code2wav_upsample_0_0
        P_code2wav_upsample_0_1["code2wav...0.1\nConvNeXtBlock\n1x1024x120->1x1024x120"]
        P_code2wav_upsample_0_0 --> P_code2wav_upsample_0_1
        P_code2wav_upsample_1_0["code2wav...1.0\nCausalTransConvNet\n1x1024x120->1x1024x240"]
        P_code2wav_upsample_0_1 --> P_code2wav_upsample_1_0
        P_code2wav_upsample_1_1["code2wav...1.1\nConvNeXtBlock\n1x1024x240->1x1024x240"]
        P_code2wav_upsample_1_0 --> P_code2wav_upsample_1_1
        P_code2wav_decoder_0_conv["code2wav...0.conv\nConv1d\n1x1024x246->1x1536x240"]
        P_code2wav_upsample_1_1 --> P_code2wav_decoder_0_conv
        P_code2wav_decoder_0["code2wav.decoder.0\nCausalConvNet\n1x1024x240->1x1536x240"]
        P_code2wav_decoder_0_conv --> P_code2wav_decoder_0
        P_code2wav_decoder_1["code2wav.decoder.1\nCode2WavDecoderBlock\n1x1536x240->1x768x1912"]
        P_code2wav_decoder_0 --> P_code2wav_decoder_1
        P_code2wav_decoder_2["code2wav.decoder.2\nCode2WavDecoderBlock\n1x768x1912->1x384x9555"]
        P_code2wav_decoder_1 --> P_code2wav_decoder_2
        P_code2wav_decoder_3["code2wav.decoder.3\nCode2WavDecoderBlock\n1x384x9555->1x192x38216"]
        P_code2wav_decoder_2 --> P_code2wav_decoder_3
        P_code2wav_decoder_4["code2wav.decoder.4\nCode2WavDecoderBlock\n1x192x38216->1x96x114645"]
        P_code2wav_decoder_3 --> P_code2wav_decoder_4
        P_code2wav_decoder_5["code2wav.decoder.5\nSnakeBeta\n1x96x114645->1x96x114645"]
        P_code2wav_decoder_4 --> P_code2wav_decoder_5
        P_code2wav_decoder_6_conv["code2wav...6.conv\nConv1d\n1x96x114651->1x1x114645"]
        P_code2wav_decoder_5 --> P_code2wav_decoder_6_conv
        P_code2wav_decoder_6["code2wav.decoder.6\nCausalConvNet\n1x96x114645->1x1x114645"]
        P_code2wav_decoder_6_conv --> P_code2wav_decoder_6
        P_code2wav["code2wav\nCode2Wav\n1x16x60->1x1x114645"]
        P_code2wav_decoder_6 --> P_code2wav
    end
```
