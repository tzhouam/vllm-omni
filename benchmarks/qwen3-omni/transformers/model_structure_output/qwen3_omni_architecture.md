# Qwen3 Omni 30B 模型架构

完整的 Qwen3 Omni 30B 模型架构,展示数据流、形状和类型。

## 完整架构流程图

```mermaid
flowchart TD
    %% Input
    Input["输入文本 Token<br/>Input:  1x14 int64<br/>Output: 1x14 int64"]
    
    %% ===== THINKER 部分 (思维模块) =====
    subgraph Thinker["🧠 Thinker (多模态理解 + 文本生成)<br/>Input:  1x14 int64<br/>Output1: 1x14x2048 bf16 (hidden)<br/>Output2: 1x14x152064 bf16 (logits)"]
        T1["Embedding Layer<br/>[1, 14] int64<br/>↓<br/>[1, 14, 2048] bf16"]
        T2["Rotary Embedding<br/>RoPE 位置编码<br/>[1, 14, 2048] bf16"]
        
        subgraph T_Layers["48 × Decoder Layer (MoE)"]
            TL1["Layer N"]
            TL1_LN1["RMSNorm<br/>[1, seq, 2048] bf16"]
            TL1_Attn["Self Attention<br/>多头注意力<br/>[1, seq, 2048] bf16"]
            TL1_LN2["RMSNorm<br/>[1, seq, 2048] bf16"]
            TL1_MoE["Sparse MoE Block<br/>专家混合层<br/>128 experts<br/>[1, seq, 2048] bf16"]
            
            TL1_LN1 --> TL1_Attn
            TL1_Attn --> TL1_LN2
            TL1_LN2 --> TL1_MoE
        end
        
        T3["Final RMSNorm<br/>[1, seq, 2048] bf16"]
        T4["LM Head (Linear)<br/>[1, seq, 2048] bf16<br/>↓<br/>[1, seq, 152064] bf16<br/>词表大小: 152064"]
        
        T1 --> T2
        T2 --> T_Layers
        T_Layers --> T3
        T3 --> T4
    end
    
    %% ===== BRIDGE 桥接层 =====
    Bridge["🔗 Text Projection (ResizeMLP)<br/>Linear(2048→2048) + SiLU → Linear(2048→1024)<br/>Input:  1x20x2048 bf16 (示例)<br/>Output: 1x20x1024 bf16 (示例)"]
    
    %% ===== TALKER 部分 (语音生成) =====
    subgraph Talker["🗣️ Talker (语音生成)<br/>Input:  1x20x1024 bf16 (示例)<br/>Output: 1x16x39 int64 (RVQ codes, 示例)"]
        direction TB
        
        subgraph Talker_LM["Talker Language Model<br/>Input:  1x20x1024 bf16 (示例)<br/>Output: 1x20x1024 bf16 (示例)"]
            TA1["Codec Embedding<br/>[1, 6] int64<br/>↓<br/>[1, 6, 1024] bf16"]
            TA2["Rotary Embedding<br/>[1, 20, 1024] bf16"]
            
            subgraph TA_Layers["48 × Decoder Layer (MoE)"]
                TAL1["Layer N"]
                TAL1_LN1["RMSNorm<br/>[1, seq, 1024] bf16"]
                TAL1_Attn["Self Attention<br/>[1, seq, 1024] bf16"]
                TAL1_LN2["RMSNorm<br/>[1, seq, 1024] bf16"]
                TAL1_MoE["Sparse MoE Block<br/>128 experts<br/>[1, seq, 1024] bf16"]
                
                TAL1_LN1 --> TAL1_Attn
                TAL1_Attn --> TAL1_LN2
                TAL1_LN2 --> TAL1_MoE
            end
            
            TA3["Final RMSNorm<br/>[1, seq, 1024] bf16"]
            
            TA1 --> TA2
            TA2 --> TA_Layers
            TA_Layers --> TA3
        end
        
        subgraph CodePredictor["Code Predictor (多码本预测)<br/>Input:  1x1x1024 bf16 (示例)<br/>Output: 1x1x2048 bf16 (logits, 示例) → sample → int64 code"]
            direction TB
            CP1["16 × Codec Heads<br/>每个码本独立预测"]
            
            subgraph CP_Single["单个 Codec Head"]
                CP_Emb["Codec Embedding<br/>[1, 1] int64<br/>↓<br/>[1, 1, 1024] bf16"]
                CP_RoPE["Rotary Embedding"]
                
                subgraph CP_Layers["5 × Decoder Layer"]
                    CPL1["Layer N"]
                    CPL1_LN1["RMSNorm"]
                    CPL1_Attn["Self Attention"]
                    CPL1_LN2["RMSNorm"]
                    CPL1_MLP["MLP"]
                    
                    CPL1_LN1 --> CPL1_Attn
                    CPL1_Attn --> CPL1_LN2
                    CPL1_LN2 --> CPL1_MLP
                end
                
                CP_Norm["RMSNorm"]
                CP_Head["LM Head (Linear)<br/>[1, 1, 1024] bf16<br/>↓<br/>[1, 1, 2048] bf16"]
                
                CP_Emb --> CP_RoPE
                CP_RoPE --> CP_Layers
                CP_Layers --> CP_Norm
                CP_Norm --> CP_Head
            end
            
            CP1 --> CP_Single
        end
        
        Talker_LM --> CodePredictor
    end
    
    %% ===== CODE2WAV 部分 (音频解码) =====
    subgraph Code2Wav["🎵 Code2Wav (Vocoder 声码器)<br/>Input:  1x16x39 int64 (RVQ codes)<br/>Output: 1xT_audio float32 (waveform)"]
        direction TB
        C1["Code Embedding<br/>[1, 16, 39] int64<br/>↓<br/>[1, 16, 39, 1024] bf16<br/>16个码本, 共39个codes"]
        
        subgraph C_PreTrans["Pre Transformer"]
            C2["Rotary Embedding<br/>[1, 39, 1024] bf16"]
            
            subgraph C_PreLayers["18 × Transformer Layer"]
                CL1["Layer N"]
                CL1_LN1["RMSNorm"]
                CL1_Attn["Self Attention"]
                CL1_Scale1["Layer Scale"]
                CL1_LN2["RMSNorm"]
                CL1_MLP["MLP"]
                CL1_Scale2["Layer Scale"]
                
                CL1_LN1 --> CL1_Attn
                CL1_Attn --> CL1_Scale1
                CL1_Scale1 --> CL1_LN2
                CL1_LN2 --> CL1_MLP
                CL1_MLP --> CL1_Scale2
            end
            
            C3["Final RMSNorm<br/>[1, 39, 1024] bf16"]
            
            C2 --> C_PreLayers
            C_PreLayers --> C3
        end
        
        subgraph C_PostTrans["Post Transformer"]
            C4["Rotary Embedding"]
            
            subgraph C_PostLayers["18 × Transformer Layer"]
                CPL["类似 Pre Transformer 结构"]
            end
            
            C5["Final RMSNorm"]
            C4 --> C_PostLayers
            C_PostLayers --> C5
        end
        
        C6["Conv Layers<br/>上采样 + 卷积"]
        C7["Wave Output<br/>音频波形<br/>[1, time_steps]<br/>float32"]
        
        C1 --> C_PreTrans
        C_PreTrans --> C_PostTrans
        C_PostTrans --> C6
        C6 --> C7
    end
    
    %% ===== 数据流连接 =====
    Input --> Thinker
    Thinker --> Bridge
    Bridge --> Talker
    Talker --> Code2Wav
    Code2Wav --> Output["音频输出<br/>Waveform"]
    
    %% 样式定义
    classDef thinkerClass fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef talkerClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef code2wavClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef bridgeClass fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef ioClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class T1,T2,T_Layers,T3,T4,TL1,TL1_LN1,TL1_Attn,TL1_LN2,TL1_MoE thinkerClass
    class TA1,TA2,TA_Layers,TA3,TAL1,TAL1_LN1,TAL1_Attn,TAL1_LN2,TAL1_MoE talkerClass
    class CP1,CP_Single,CP_Emb,CP_RoPE,CP_Layers,CP_Norm,CP_Head,CPL1,CPL1_LN1,CPL1_Attn,CPL1_LN2,CPL1_MLP talkerClass
    class C1,C2,C_PreLayers,C3,C4,C_PostLayers,C5,C6,C7,CL1,CL1_LN1,CL1_Attn,CL1_Scale1,CL1_LN2,CL1_MLP,CL1_Scale2,CPL code2wavClass
    class Bridge bridgeClass
    class Input,Output ioClass
```

## 模型详细规格

### 1. Thinker (思维模块)
- **功能**: 多模态理解和文本生成
- **层数**: 48 层 Transformer Decoder
- **隐藏维度**: 2048
- **MoE 配置**: 
  - 每层 128 个专家
  - 稀疏激活(只激活部分专家)
- **输入**: 文本 token (int64)
- **输出**: 文本logits [vocab_size=152064]
- **数据类型**: bfloat16

**每层结构**:
```
输入 → RMSNorm → Self-Attention → 残差连接
     ↓
     → RMSNorm → Sparse MoE Block → 残差连接 → 输出
```

### 2. Text Projection (桥接层)
- **功能**: 将 Thinker 的 2048 维降到 Talker 的 1024 维
- **结构**: 
  - Linear(2048 → 2048)
  - SiLU 激活
  - Linear(2048 → 1024)
- **数据类型**: bfloat16

### 3. Talker (语音生成模块)

#### 3.1 Talker Language Model
- **层数**: 48 层 Transformer Decoder
- **隐藏维度**: 1024
- **MoE 配置**: 每层 128 个专家
- **输入**: 来自 Thinker 的投影 + Codec embeddings
- **输出**: 中间表示用于 Code Predictor

#### 3.2 Code Predictor (码本预测器)
- **功能**: 多码本预测(Multi-Token Prediction)
- **码本数量**: 16 个独立码本
- **每个码本**:
  - 5 层 Transformer Decoder
  - 隐藏维度: 1024
  - 输出: 2048 个 codes
- **数据类型**: bfloat16

### 4. Code2Wav (声码器)
- **功能**: 将 RVQ codes 转换为音频波形
- **输入**: 16 个码本 × 39 个 codes/帧
- **结构**:
  - **Pre Transformer**: 18 层
    - RMSNorm + Self-Attention + Layer Scale
    - RMSNorm + MLP + Layer Scale
  - **Post Transformer**: 18 层(相同结构)
  - **Wave Conv**: 上采样卷积层
- **输出**: 音频波形 (float32)
- **数据类型**: bfloat16 (transformer), float32 (output)

## 关键特性

### MoE (Mixture of Experts)
- **专家数量**: 128 个专家/层
- **激活策略**: 稀疏激活
- **位置**: Thinker 和 Talker 的每个 MLP 层

### 注意力机制
- **类型**: Multi-Head Self-Attention
- **位置编码**: RoPE (Rotary Position Embedding)
- **KV Cache**: 支持增量解码

### 归一化
- **类型**: RMSNorm (Root Mean Square Layer Normalization)
- **位置**: 每个 attention/MLP 之前

### 数据类型优化
- **主要计算**: bfloat16 (降低显存,加速计算)
- **输入 tokens**: int64
- **音频输出**: float32 (保证质量)

## 数据流总览

```
文本输入 (int64)
    ↓
Thinker Embedding (2048-dim, bf16)
    ↓
48 × Decoder Layer (MoE)
    ↓
LM Head → 文本 logits
    ↓
Text Projection (2048→1024)
    ↓
Talker LM (48 × Decoder Layer, MoE)
    ↓
Code Predictor (16 × 5-layer, 预测 RVQ codes)
    ↓
Code2Wav Pre-Transformer (18层)
    ↓
Code2Wav Post-Transformer (18层)
    ↓
Wave Conv (上采样)
    ↓
音频波形输出 (float32)
```

## 模型规模

- **总参数**: ~30B
- **Thinker**: ~20B (48层 × 2048-dim MoE)
- **Talker**: ~8B (48层 × 1024-dim MoE + Code Predictor)
- **Code2Wav**: ~2B (36层 transformer + conv)

## 主要优化

1. **MoE 架构**: 大容量参数,实际激活参数较少
2. **bfloat16**: 降低显存占用,加速训练和推理
3. **多阶段设计**: Thinker→Talker→Code2Wav 流水线
4. **多码本预测**: 16个码本并行预测,提高音频质量

---

*生成时间: 2026-01-09*
*基于 Qwen3-Omni-30B-A3B-Thinking 模型*
