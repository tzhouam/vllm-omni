# Qwen3 Omni 层级细节

详细展示各组件的内部结构和操作。

## Thinker Decoder Layer 详细结构

```mermaid
flowchart TD
    Input["层输入<br/>[B, L, 2048] bf16"]
    
    %% Attention Branch
    subgraph AttentionBranch["Attention 分支"]
        LN1["RMSNorm<br/>━━━━━━━━━━<br/>γ: [2048]<br/>[B,L,2048]→[B,L,2048]"]
        
        subgraph SelfAttn["Multi-Head Self-Attention"]
            Q["Q Projection<br/>Linear(2048→2048)<br/>bf16"]
            K["K Projection<br/>Linear(2048→512)<br/>bf16"]
            V["V Projection<br/>Linear(2048→512)<br/>bf16"]
            RoPE["RoPE<br/>旋转位置编码<br/>cos/sin table"]
            Attn["Scaled Dot-Product<br/>━━━━━━━━━━<br/>softmax(QK^T/√d)V<br/>16 heads"]
            O["O Projection<br/>Linear(2048→2048)<br/>bf16"]
            
            Q --> RoPE
            K --> RoPE
            RoPE --> Attn
            V --> Attn
            Attn --> O
        end
        
        LN1 --> SelfAttn
    end
    
    Residual1["➕ Residual<br/>Add & Normalize"]
    
    %% MoE Branch
    subgraph MoEBranch["MoE 分支"]
        LN2["RMSNorm<br/>━━━━━━━━━━<br/>[B,L,2048]→[B,L,2048]"]
        
        subgraph MoE["Sparse MoE Block"]
            Gate["Router/Gate<br/>━━━━━━━━━━<br/>Linear(2048→128)<br/>softmax → top-k<br/>选择激活专家"]
            
            subgraph Experts["128 个专家 (并行)"]
                E1["Expert 1<br/>━━━━━━━━━━<br/>Gate: Linear(2048→11008)<br/>Activation: SwiGLU<br/>Up: Linear(11008→2048)"]
                E2["Expert 2<br/>(同样结构)"]
                Edot["..."]
                E128["Expert 128<br/>(同样结构)"]
                
                E1 -.-> Edot
                Edot -.-> E128
            end
            
            Combine["加权组合<br/>━━━━━━━━━━<br/>Σ gate_weight_i × expert_i"]
            
            Gate --> Experts
            Experts --> Combine
        end
        
        SharedExpert["Shared Expert<br/>(总是激活)<br/>同样的 FFN 结构"]
        
        LN2 --> MoE
        MoE --> SharedExpert
    end
    
    Residual2["➕ Residual<br/>Add & Normalize"]
    Output["层输出<br/>[B, L, 2048] bf16"]
    
    %% Flow
    Input --> AttentionBranch
    AttentionBranch --> Residual1
    Input -.->|shortcut| Residual1
    Residual1 --> MoEBranch
    MoEBranch --> Residual2
    Residual1 -.->|shortcut| Residual2
    Residual2 --> Output
    
    classDef norm fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef linear fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef attn fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef moe fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef res fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    
    class LN1,LN2 norm
    class Q,K,V,O,Gate linear
    class RoPE,Attn,SelfAttn attn
    class MoE,Experts,E1,E2,E128,SharedExpert moe
    class Residual1,Residual2 res
```

## 详细操作说明

### 1. RMSNorm (Root Mean Square Normalization)

```python
def rms_norm(x, weight, eps=1e-6):
    """
    x: [B, L, D]
    weight: [D]
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x
```

**输入/输出**:
- 输入: `[batch, seq_len, hidden]` bfloat16
- 输出: `[batch, seq_len, hidden]` bfloat16

### 2. Self-Attention (自注意力)

```python
def self_attention(x, W_q, W_k, W_v, W_o, rope):
    """
    x: [B, L, D]
    W_q: [D, D], W_k: [D, D_kv], W_v: [D, D_kv], W_o: [D, D]
    """
    Q = x @ W_q  # [B, L, D]
    K = x @ W_k  # [B, L, D_kv]
    V = x @ W_v  # [B, L, D_kv]
    
    # Apply RoPE
    Q, K = rope(Q, K)
    
    # Multi-head attention
    Q = Q.view(B, L, num_heads, head_dim).transpose(1, 2)  # [B, H, L, D/H]
    K = K.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    V = V.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    
    # Grouped Query Attention (GQA)
    attn = softmax(Q @ K.T / sqrt(head_dim))  # [B, H, L, L]
    out = attn @ V  # [B, H, L, D/H]
    
    # Concatenate heads
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    return out @ W_o
```

**参数**:
- Thinker: 16 heads, head_dim=128, GQA ratio=4
- Talker: 16 heads, head_dim=64, GQA ratio=4

**形状**:
- Q: `[B, L, 2048]` → `[B, 16, L, 128]`
- K,V: `[B, L, 512]` → `[B, 4, L, 128]`

### 3. RoPE (Rotary Position Embedding)

```python
def apply_rotary_emb(q, k, cos, sin):
    """
    Rotate half of the dimensions
    """
    q_rot = rotate_half(q)
    k_rot = rotate_half(k)
    
    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    return q_out, k_out

def rotate_half(x):
    """Rotate half of hidden dims"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
```

### 4. Sparse MoE (稀疏专家混合)

```python
def sparse_moe(x, gate, experts, k=2):
    """
    x: [B, L, D]
    gate: [D, num_experts]
    experts: list of FFN modules
    k: top-k experts to activate
    """
    # Router
    router_logits = x @ gate  # [B, L, num_experts]
    router_probs = softmax(router_logits, dim=-1)
    
    # Select top-k experts
    top_k_probs, top_k_indices = torch.topk(router_probs, k, dim=-1)
    # [B, L, k]
    
    # Normalize weights
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    # Expert computation (only compute selected experts)
    output = torch.zeros_like(x)
    for i in range(k):
        expert_idx = top_k_indices[:, :, i]  # [B, L]
        expert_weight = top_k_probs[:, :, i:i+1]  # [B, L, 1]
        
        # Batch expert calls
        expert_out = experts[expert_idx](x)  # [B, L, D]
        output += expert_weight * expert_out
    
    return output
```

**Thinker MoE 配置**:
- 专家数: 128
- Top-k: 2 (每次激活2个专家)
- Shared expert: 1 (总是激活)
- Expert FFN: 2048 → 11008 → 2048

**Talker MoE 配置**:
- 专家数: 128
- Top-k: 2
- Shared expert: 1
- Expert FFN: 1024 → 5504 → 1024

### 5. FFN (Feed-Forward Network) 在专家中

```python
def ffn_expert(x, W_gate, W_up, W_down):
    """
    SwiGLU activation: swish(xW_gate) ⊙ (xW_up)
    
    x: [B, L, D]
    W_gate: [D, D_ff]
    W_up: [D, D_ff]
    W_down: [D_ff, D]
    """
    gate = F.silu(x @ W_gate)  # [B, L, D_ff]
    up = x @ W_up              # [B, L, D_ff]
    return (gate * up) @ W_down  # [B, L, D]
```

## Talker Decoder Layer

与 Thinker 类似,但有以下差异:

```mermaid
flowchart TD
    Input["层输入<br/>[B, L, 1024] bf16"]
    LN1["RMSNorm"]
    Attn["Self-Attention<br/>━━━━━━━━━━<br/>16 heads, head_dim=64<br/>Q: 1024→1024<br/>K,V: 1024→256"]
    Res1["➕ Residual"]
    LN2["RMSNorm"]
    MoE["Sparse MoE<br/>━━━━━━━━━━<br/>128 experts<br/>FFN: 1024→5504→1024<br/>top-k=2"]
    Res2["➕ Residual"]
    Output["层输出<br/>[B, L, 1024] bf16"]
    
    Input --> LN1 --> Attn --> Res1
    Input -.-> Res1
    Res1 --> LN2 --> MoE --> Res2
    Res1 -.-> Res2
    Res2 --> Output
```

**关键差异**:
- Hidden size: 2048 → 1024
- Head dim: 128 → 64
- FFN intermediate: 11008 → 5504

## Code Predictor Layer

```mermaid
flowchart TD
    Input["层输入<br/>[B, L, 1024] bf16"]
    LN1["RMSNorm"]
    Attn["Self-Attention<br/>━━━━━━━━━━<br/>8 heads, head_dim=128"]
    Res1["➕ Residual"]
    LN2["RMSNorm"]
    MLP["MLP (非 MoE)<br/>━━━━━━━━━━<br/>1024→2816→1024<br/>SwiGLU"]
    Res2["➕ Residual"]
    Output["层输出<br/>[B, L, 1024] bf16"]
    
    Input --> LN1 --> Attn --> Res1
    Input -.-> Res1
    Res1 --> LN2 --> MLP --> Res2
    Res1 -.-> Res2
    Res2 --> Output
```

**关键特点**:
- **无 MoE**: 使用普通 MLP
- **更小的网络**: 5 层 vs 48 层
- **独立码本**: 16 个预测器并行

## Code2Wav Transformer Layer

```mermaid
flowchart TD
    Input["层输入<br/>[B, L, 1024] bf16"]
    LN1["RMSNorm"]
    Attn["Self-Attention<br/>━━━━━━━━━━<br/>16 heads"]
    Scale1["Layer Scale<br/>━━━━━━━━━━<br/>可学习缩放因子<br/>初始值≈0"]
    Res1["➕ Residual"]
    LN2["RMSNorm"]
    MLP["MLP<br/>━━━━━━━━━━<br/>1024→4096→1024<br/>GELU"]
    Scale2["Layer Scale"]
    Res2["➕ Residual"]
    Output["层输出<br/>[B, L, 1024] bf16"]
    
    Input --> LN1 --> Attn --> Scale1 --> Res1
    Input -.-> Res1
    Res1 --> LN2 --> MLP --> Scale2 --> Res2
    Res1 -.-> Res2
    Res2 --> Output
```

**关键特点**:
- **Layer Scale**: 稳定深层网络训练
- **无 MoE**: 所有层都是密集 MLP
- **GELU 激活**: 代替 SwiGLU

## 计算复杂度分析

### Thinker Decoder Layer

| 操作 | 参数量 | 计算量 (FLOPs) |
|------|--------|----------------|
| RMSNorm × 2 | 2×2048 | 忽略不计 |
| Q Projection | 2048×2048 | 2BLD² |
| K Projection | 2048×512 | 0.5BLD² |
| V Projection | 2048×512 | 0.5BLD² |
| Attention | - | 2BL²D |
| O Projection | 2048×2048 | 2BLD² |
| MoE Router | 2048×128 | 忽略不计 |
| Expert (×2激活) | 2×(2048×11008×2) | 8BLD_ff |
| Shared Expert | 2048×11008×2 | 4BLD_ff |
| **总计** | ~47M/layer | ~12BLD_ff |

其中:
- B = batch size
- L = sequence length  
- D = 2048
- D_ff = 11008

### 总参数量估算

```
Thinker: 48 layers × 47M ≈ 2.3B (骨干)
         + Embedding (152K × 2048) ≈ 311M
         + MoE experts (128 × 48 × 90M) ≈ 18B
         = ~20.6B

Talker:  48 layers × 12M ≈ 576M (骨干)
         + MoE experts (128 × 48 × 22M) ≈ 7B
         + Code Predictor (16 × 5 × 3M) ≈ 240M
         = ~7.8B

Code2Wav: 36 layers × 13M ≈ 470M
          + Conv layers ≈ 30M
          = ~500M

Total: 20.6B + 7.8B + 0.5B ≈ 29B
```

## 数据类型与精度

### bfloat16 格式
```
符号位(1) | 指数位(8) | 尾数位(7)
    ↓          ↓           ↓
   [0]    [11111110]   [1111111]

表示范围: ±3.4×10^38 (同 float32)
精度: ~3位十进制 (vs float32的7位)
```

**优势**:
- ✅ 显存占用减半
- ✅ 硬件加速(TPU/GPU原生支持)
- ✅ 动态范围大(指数位同 float32)
- ✅ 训练稳定(不易溢出)

**劣势**:
- ❌ 精度降低(7位尾数 vs 23位)
- ❌ 小数表示不精确

---

*详细层级分析 - Qwen3 Omni 30B*
