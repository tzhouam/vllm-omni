# Qwen3 Omni 30B 模型架构文档

完整的 Qwen3 Omni 30B 模型架构可视化和分析文档。

## 📚 文档索引

### 🎯 推荐阅读顺序

1. **[高层架构](./qwen3_omni_highlevel.md)** ⭐ 入门必读
   - 简洁的组件概览
   - 主要数据流
   - 关键技术点
   - **适合**: 快速了解模型整体结构

2. **[完整架构](./qwen3_omni_architecture.md)** ⭐⭐ 深入理解
   - 详细的流程图
   - 完整的数据流转
   - 各组件详细规格
   - **适合**: 需要实现或优化模型

3. **[层级细节](./qwen3_omni_layer_details.md)** ⭐⭐⭐ 专家级
   - 单层内部结构
   - 具体操作和公式
   - 参数量计算
   - 复杂度分析
   - **适合**: 深度优化和研究

### 📊 自动生成的图表

#### 按组件分类

- **[Thinker 完整概览](./qwen3_omni_thinker_overall_mermaid.md)**
  - Thinker 全部 48 层的完整结构
  
- **[Thinker 基础模型](./qwen3_omni_thinker_mermaid.md)**
  - 非 MoE 部分的基础结构
  
- **[Thinker MoE 详解](./qwen3_omni_thinker_moe_mermaid.md)**
  - 专家混合层的详细结构

- **[Talker 完整概览](./qwen3_omni_talker_overall_mermaid.md)**
  - Talker 全部结构概览
  
- **[Talker 语言模型](./qwen3_omni_talker_lm_mermaid.md)**
  - Talker 的 48 层 transformer
  
- **[Talker MoE](./qwen3_omni_talker_moe_mermaid.md)**
  - Talker 中的 MoE 层
  
- **[Talker MTP](./qwen3_omni_talker_mtp_mermaid.md)**
  - Multi-Token Prediction (码本预测器)

- **[Code2Wav 声码器](./qwen3_omni_code2wav_mermaid.md)**
  - 音频生成网络结构

#### 原始数据

- **[层详细信息表](./qwen3_omni_layers.md)**
  - 所有 15,222 层的详细参数
  - 输入/输出形状和数据类型
  - Markdown 表格格式

- **[原始 JSON 数据](./qwen3_omni_data.json)**
  - 机器可读的完整数据
  - 适合程序分析和处理

## 🏗️ 模型架构概览

```
Qwen3 Omni 30B
├── Thinker (思维模块) ~20B
│   ├── Embedding Layer
│   ├── 48 × MoE Decoder Layer
│   │   ├── Self-Attention (GQA)
│   │   └── Sparse MoE (128 experts)
│   └── LM Head
│
├── Text Projection (桥接) ~10M
│   └── MLP (2048→1024)
│
├── Talker (语音生成) ~8B
│   ├── Language Model (48 × MoE Decoder)
│   └── Code Predictor (16 × 5-layer)
│       └── 16 个独立码本预测器
│
└── Code2Wav (声码器) ~2B
    ├── Pre-Transformer (18 layers)
    ├── Post-Transformer (18 layers)
    └── Wave Conv (上采样)
```

## 🔑 关键技术

### MoE (Mixture of Experts)
- **位置**: Thinker 和 Talker 的 MLP 层
- **专家数**: 每层 128 个
- **激活策略**: Top-2 + 1 shared expert
- **优势**: 大容量参数,低计算成本

### RoPE (Rotary Position Embedding)
- **作用**: 位置编码
- **优势**: 支持外推,长序列性能好

### GQA (Grouped Query Attention)
- **结构**: Q 16 heads, K/V 4 heads
- **优势**: 降低 KV Cache 大小

### RVQ (Residual Vector Quantization)
- **码本数**: 16 个
- **作用**: 音频压缩表示
- **优势**: 高质量,渐进式细化

### bfloat16
- **应用**: 主要计算
- **优势**: 显存减半,速度提升

## 📏 模型规格

| 组件 | 层数 | 隐藏维度 | 参数量 | MoE | 数据类型 |
|------|------|----------|--------|-----|----------|
| **Thinker** | 48 | 2048 | ~20B | ✅ | bfloat16 |
| **Talker LM** | 48 | 1024 | ~7B | ✅ | bfloat16 |
| **Code Predictor** | 16×5 | 1024 | ~1B | ❌ | bfloat16 |
| **Code2Wav** | 36 | 1024 | ~2B | ❌ | bf16→f32 |
| **总计** | - | - | **~30B** | - | - |

## 🎵 数据流

```
文本输入 (Token IDs)
    ↓ [B, L] int64
Thinker Embedding
    ↓ [B, L, 2048] bfloat16
Thinker Layers (48×)
    ↓ [B, L, 2048] bfloat16
Text Projection
    ↓ [B, L, 1024] bfloat16
Talker LM (48×)
    ↓ [B, L, 1024] bfloat16
Code Predictor (16×)
    ↓ [B, 16, T] int64 (RVQ codes)
Code2Wav Embedding
    ↓ [B, 16, T, 1024] bfloat16
Pre+Post Transformer (36×)
    ↓ [B, T, 1024] bfloat16
Wave Conv
    ↓ [B, audio_len] float32
音频输出
```

## 💡 使用建议

### 快速查看
```bash
# 查看高层架构
cat qwen3_omni_highlevel.md

# 查看特定组件
cat qwen3_omni_thinker_mermaid.md
```

### 在 Markdown 查看器中渲染
支持 Mermaid 的工具:
- GitHub (自动渲染)
- VS Code (Markdown Preview Mermaid 插件)
- Typora
- Obsidian
- GitLab

### 程序化分析
```python
import json

# 读取原始数据
with open('qwen3_omni_data.json') as f:
    data = json.load(f)

# 查看总览
print(data['summary'])
# {'total_layers': 15222, 'total_hooks': 911}

# 分析特定组件
thinker_layers = [
    layer for layer in data['layers']
    if layer['subcomponent'] == 'thinker'
]
print(f"Thinker layers: {len(thinker_layers)}")
```

## 🔧 生成这些文档

使用 `generate_model_structure.py` 脚本:

```bash
python generate_model_structure.py \
    --model_path /path/to/Qwen3-Omni-30B \
    --output_dir ./model_structure_output \
    --split_by_component \
    --max_edges 499 \
    --max_nodes 200 \
    --max_layer_depth 6
```

**参数说明**:
- `--split_by_component`: 按组件分别生成图表
- `--max_edges 499`: 每个图最多 499 条边
- `--max_nodes 200`: 每个图最多 200 个节点
- `--max_layer_depth 6`: 最大层级深度
- `--no_compact`: 禁用紧凑模式(显示完整名称)

## 📖 相关资源

### 官方资源
- [Qwen3-Omni 技术报告](https://arxiv.org/abs/2024.xxxxx)
- [Hugging Face 模型](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [GitHub 仓库](https://github.com/QwenLM/Qwen3-Omni)

### 参考论文
- **MoE**: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- **RoPE**: [RoFormer](https://arxiv.org/abs/2104.09864)
- **GQA**: [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- **RVQ**: [SoundStream](https://arxiv.org/abs/2107.03312)

## 🤝 贡献

这些文档是自动生成的,如果发现错误或有改进建议:

1. 修改 `qwen3_omni_moe_transformers.py` 中的 tracing 逻辑
2. 修改 `generate_model_structure.py` 中的生成逻辑
3. 重新运行生成脚本

## 📝 更新日志

- **2026-01-09**: 初始版本
  - 创建完整架构文档
  - 添加高层概览
  - 添加层级细节分析
  - 按组件分类生成图表

---

**模型**: Qwen3-Omni-30B-A3B-Thinking  
**生成时间**: 2026-01-09  
**文档版本**: 1.0
