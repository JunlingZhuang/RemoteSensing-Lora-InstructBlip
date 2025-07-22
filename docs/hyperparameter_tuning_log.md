# LoRA InstructBLIP 超参数调参记录

## 📋 项目概述

- **模型**: InstructBLIP + LoRA 微调
- **数据集**: RSICap (遥感图像描述)
- **目标**: 优化训练稳定性和收敛效果

---

## 🔧 调参历程

### 实验 1: 初始配置 (失败 - NaN 问题)

```python
TRAINING_CONFIG = {
    "num_epochs": 10,
    "batch_size": 2,
    "learning_rate": 5e-6,
    "max_samples": 20,
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "torch_dtype": "float16"  # 问题根源！
}
```

**结果**:

- ❌ 第 1 个 batch 成功，后续全部 NaN
- ❌ Forward 返回 None，训练无法继续

**问题分析**:

- **根本原因**: `float16` 数值精度不足
- **现象**: LoRA 参数更新后数值不稳定
- **解决方案**: 切换到 `float32`

---

### 实验 2: 修复数值稳定性 (成功)

```python
TRAINING_CONFIG = {
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 5e-6,
    "max_samples": 400,
    "lora_r": 8,
    "lora_alpha": 16,
    "torch_dtype": "float32"  # 关键修复！
}
```

**结果**:

- ✅ 训练稳定，无 NaN 问题
- ✅ Loss 正常下降: 2.06 → 1.55 (训练), 1.74 → 1.50 (验证)
- ⚠️ 下降幅度不够大

**学到的经验**:

1. **float32 vs float16**: LoRA 微调对数值精度敏感
2. **自定义 forward 必要性**: HF 原生 forward 与 LoRA 冲突
3. **梯度裁剪重要性**: 防止参数爆炸

---

### 实验 3: 激进参数 (过拟合)

```python
TRAINING_CONFIG = {
    "num_epochs": 15,
    "batch_size": 4,
    "learning_rate": 1e-3,    # 太高！
    "max_samples": 400,
    "lora_r": 16,             # 太大！
    "lora_alpha": 32,         # 太大！
    "lora_dropout": 0.05,
    "torch_dtype": "float32"
}
```

**结果**:

- ✅ 前 3 个 epoch 良好下降: 1.89 → 1.71 → 1.69
- ❌ 第 4 个 epoch 开始上升: 1.69 → 1.87
- ❌ 验证 loss 也在第 3 个 epoch 开始上升

**问题分析**:

1. **学习率过高**: 1e-3 对 LoRA 太激进，导致震荡
2. **LoRA 容量过大**: r=16 可能给模型太多自由度
3. **过拟合迹象**: 训练和验证 loss 同时上升

---

### 实验 4: 平衡配置 (已完成)

```python
TRAINING_CONFIG = {
    "num_epochs": 15,
    "batch_size": 4,
    "learning_rate": 2e-4,    # 降低学习率
    "max_samples": 400,
    "lora_r": 16,             # 保持较高表达能力
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_steps": 200,
    "max_grad_norm": 1.0,
    "torch_dtype": "float32"
}
```

**结果**: 待补充训练完成后的数据...

---

### 实验 5: 扩大数据集 (性能问题)

```python
TRAINING_CONFIG = {
    "num_epochs": 15,
    "batch_size": 8,          # 增加batch size，更稳定梯度
    "learning_rate": 2e-4,    # 保持平衡的学习率
    "max_samples": 100,      # 大幅增加训练数据！
    "lora_r": 16,             # 保持较高表达能力
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_steps": 200,
    "max_grad_norm": 1.0,
    "torch_dtype": "float32"
}
```

**结果**: ❌ 性能问题 - 20 分钟/epoch，太慢！

---

### 实验 5b: 性能优化 (当前)

```python
TRAINING_CONFIG = {
    "num_epochs": 15,
    "batch_size": 16,         # 8→16，减少batch数量，加速训练
    "learning_rate": 2e-4,    # 保持平衡的学习率
    "max_samples": 2500,      # 保持大数据集
    "lora_r": 16,             # 保持较高表达能力
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_steps": 200,
    "max_grad_norm": 1.0,
    "torch_dtype": "float32",
    "num_workers": 4          # 数据加载并行化
}
```

**优化策略**:

- 增加 batch size: 312 batches → 156 batches (减半)
- 数据加载并行: num_workers=4
- 预取和持久化 worker

**预期**:

- 训练时间: 20 分钟 → ~10 分钟/epoch
- 保持数据集规模优势
- 更稳定的梯度 (更大 batch)

---

## 📊 关键发现

### 1. 数值稳定性

- **float16**: 不适合 LoRA 微调，容易 NaN
- **float32**: 必需，确保数值稳定性
- **梯度裁剪**: 防止参数爆炸的重要保障

### 2. 学习率敏感性

| 学习率 | 现象                 | 适用性   |
| ------ | -------------------- | -------- |
| 5e-6   | 收敛慢，下降不明显   | 太保守   |
| 5e-4   | 收敛快，但可能不稳定 | 中等     |
| 1e-3   | 快速下降后震荡       | 太激进   |
| 2e-4   | 待验证               | 可能最优 |

### 3. LoRA 配置权衡

- **低 rank (r=4-8)**: 稳定但表达能力有限
- **高 rank (r=16+)**: 表达能力强但可能过拟合
- **Alpha 比例**: 通常设为 rank 的 2 倍

### 4. 训练策略

- **Warmup**: 对稳定训练很重要
- **小 batch size**: 增加更新频率，有助收敛
- **早停**: 监控验证 loss，防止过拟合

---

## 🎯 最佳实践总结

### 推荐起始配置:

```python
TRAINING_CONFIG = {
    "learning_rate": 2e-4,     # 平衡收敛速度和稳定性
    "lora_r": 8,               # 适中的表达能力
    "lora_alpha": 16,          # r的2倍
    "lora_dropout": 0.1,       # 适度正则化
    "batch_size": 4,           # 小batch，频繁更新
    "warmup_steps": 100,       # 稳定启动
    "max_grad_norm": 1.0,      # 梯度裁剪
    "torch_dtype": "float32"   # 必需！
}
```

### 调参策略:

1. **先确保稳定性** (float32, 梯度裁剪)
2. **再优化收敛速度** (学习率, warmup)
3. **最后调整模型容量** (LoRA rank/alpha)
4. **持续监控过拟合** (验证 loss)

---

## 📈 监控指标

### 健康训练的特征:

- ✅ 训练 loss 平稳下降
- ✅ 验证 loss 跟随下降
- ✅ 无 NaN/Inf 出现
- ✅ 梯度范数稳定

### 问题信号:

- ❌ 训练 loss 震荡或上升
- ❌ 验证 loss 开始上升 (过拟合)
- ❌ 梯度爆炸 (需要裁剪)
- ❌ 收敛过慢 (学习率太低)

---

## 🔄 下一步计划

1. **验证当前配置** (lr=2e-4)
2. **实现学习率调度** (cosine/linear decay)
3. **尝试更大数据集** (扩展到全量数据)
4. **对比不同 LoRA 配置** (系统性实验)

---

## 🧪 技术细节

### LoRA 实现要点

```python
# 关键配置
lora_config = LoraConfig(
    r=config.lora_r,                    # 低秩分解的秩
    lora_alpha=config.lora_alpha,       # 缩放因子
    lora_dropout=config.lora_dropout,   # 正则化
    target_modules={"query", "value", "dense", "key"},  # 目标模块
    task_type=TaskType.CAUSAL_LM        # 任务类型
)
```

### 自定义 Forward 的必要性

- **HF 冲突**: 原生 InstructBLIP forward 与 LoRA 有`inputs_embeds`冲突
- **解决方案**: 手动实现 Vision → Q-Former → Language Model 流程
- **关键点**: 确保 LoRA 参数在 Q-Former 和 Language Model 中正确应用

### 数值稳定性技巧

```python
# 关键检查点
if torch.any(torch.isnan(query_embeds)) or torch.any(torch.isinf(query_embeds)):
    return None  # 跳过异常batch

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# 使用float32
config.torch_dtype = "float32"
```

---

## 📊 实验数据记录

### 实验 1 结果 (float16 + NaN 问题)

```
Batch 1: ✅ Loss: 2.386719
Batch 2: ❌ Forward returned None
Batch 3: ❌ Forward returned None
...
```

### 实验 2 结果 (float32 稳定)

```
Epoch 1: Train=2.06, Val=1.74
Epoch 2: Train=1.78, Val=1.61
...
Epoch 10: Train=1.55, Val=1.50
```

### 实验 3 结果 (学习率过高)

```
Epoch 1: Train=1.89, Val=1.63 ✅
Epoch 2: Train=1.71, Val=1.57 ✅
Epoch 3: Train=1.69, Val=1.60 ⚠️ (val开始上升)
Epoch 4: Train=1.87, Val=? ❌ (train上升)
```

### 实验 5 预期 (扩大数据集)

```
数据规模: 400 → 2500 samples (+525%)
Batch数量: 100 → 312 batches (+212%)
训练时间: ~75s → ~300s+ per epoch
预期更好的泛化能力和更稳定的收敛
```

---

## 🔍 调参决策树

```
开始训练
    ↓
是否出现NaN?
    ├─ 是 → 检查数据类型 (float32) + 梯度裁剪
    └─ 否 → 继续
         ↓
Loss是否下降?
    ├─ 否 → 提高学习率 / 增加LoRA rank
    └─ 是 → 继续
         ↓
是否震荡/过拟合?
    ├─ 是 → 降低学习率 / 增加正则化
    └─ 否 → 当前配置良好
```

---

## 📝 经验教训

### 1. 调参顺序很重要

1. **稳定性第一**: 解决 NaN 问题
2. **收敛性其次**: 确保 loss 下降
3. **优化最后**: 微调获得最佳性能

### 2. 监控多个指标

- 不只看训练 loss，验证 loss 更重要
- 学习率变化曲线有助诊断
- 梯度范数可以发现数值问题

### 3. 保守策略更可靠

- 宁可收敛慢也不要不稳定
- 小步快跑，逐步调优
- 记录每次实验的完整配置

---

_最后更新: 2025-01-22_
_下次更新: 实验 4 结果分析_
