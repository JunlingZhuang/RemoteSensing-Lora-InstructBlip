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

## 🔄 48GB GPU Grid Search 实验计划

### 实验设计概述

基于前期调参经验，设计了针对 48GB GPU 的系统性超参数搜索实验。目标是充分利用大显存优势，探索最优的训练配置。

### Grid Search 实验矩阵 (更新后 - Post OOM)

| 实验版本           | Batch Size | LoRA Rank | Learning Rate | Epochs | 策略重点           | 状态      |
| ------------------ | ---------- | --------- | ------------- | ------ | ------------------ | --------- |
| ~~V1~~             | ~~32~~     | ~~8~~     | ~~0.001~~     | ~~6~~  | ~~大批量高效训练~~ | ❌ 跳过   |
| V2                 | 10         | 16        | 0.0005        | 8      | 高容量适中批量     | ⚠️ 测试   |
| ~~V3~~             | ~~48~~     | ~~8~~     | ~~0.0008~~    | ~~4~~  | ~~极限批量测试~~   | ❌ 跳过   |
| V4                 | 8          | 32        | 0.0002        | 10     | 极限模型容量       | ⚠️ 测试   |
| V5 (新)            | 6          | 16        | 0.0008        | 12     | 内存优化配置       | ✅ 可行   |
| Ultra Conservative | 12         | 8         | 0.0003        | 10     | 紧急保守配置       | 🔄 测试中 |

### 实验配置详情

#### Grid Search V1: 大批量高效训练

```yaml
name: "grid_v1_large_batch_high_lr"
batch_size: 32 # 3x基线，充分利用48GB
learning_rate: 0.001 # 高LR配合大批量
lora_r: 8 # 保守的rank
max_samples: 2068 # 全量训练集
warmup_steps: 100 # 适中预热
```

**假设**: 大批量能提供更稳定的梯度，支持更高学习率，加速收敛。

#### Grid Search V2: 高 LoRA 秩实验

```yaml
name: "grid_v2_medium_batch_high_rank"
batch_size: 20 # 中等批量
learning_rate: 0.0005 # 基线LR
lora_r: 16 # 2x基线rank
lora_alpha: 32 # 比例缩放
lora_dropout: 0.05 # 降低dropout
warmup_steps: 150 # 延长预热
```

**假设**: 更高的 LoRA rank 提供更强的表达能力，适合复杂的遥感 VQA 任务。

#### Grid Search V3: 极限批量测试

```yaml
name: "grid_v3_ultra_batch_conservative"
batch_size: 48 # 测试48GB极限
learning_rate: 0.0008 # 适中偏高LR
lora_r: 8 # 保守rank
epochs: 4 # 减少epoch
warmup_steps: 200 # 大批量需要长预热
start_factor: 0.05 # 非常保守的预热
```

**假设**: 极大批量可能达到更好的收敛效果，但需要保守的其他参数。

#### Grid Search V4: 极限模型容量

```yaml
name: "grid_v4_extreme_rank"
batch_size: 16 # 较小批量适应高rank
learning_rate: 0.0002 # 低LR确保稳定
lora_r: 32 # 4x基线rank
lora_alpha: 64 # 比例缩放
lora_dropout: 0.02 # 极低dropout
epochs: 10 # 更多epoch
warmup_steps: 250 # 长预热
```

**假设**: 极高的模型容量能学习更复杂的视觉-语言映射，但需要更保守的训练策略。

#### Conservative Optimal: 稳妥最优

```yaml
name: "conservative_optimal_lora_instructblip"
batch_size: 24 # 安全的大批量
learning_rate: 0.0005 # 验证过的LR
lora_r: 12 # 容量与稳定性平衡点
lora_alpha: 24 # 比例缩放
lora_dropout: 0.08 # 适度正则化
epochs: 8 # 充分训练
```

**假设**: 基于前期经验的最佳平衡配置，预期效果最稳定。

### 实验执行计划

1. **Phase 1**: 运行 Conservative Optimal，建立稳定基线
2. **Phase 2**: 并行运行 Grid Search V1-V4
3. **Phase 3**: 分析结果，识别最优配置
4. **Phase 4**: 基于最佳配置进行精细调优

### 评估指标

- **训练稳定性**: 是否出现 NaN，loss 曲线平滑度
- **收敛速度**: 达到目标 loss 所需 epoch 数
- **最终性能**: RSIEval VQA 准确率
- **资源效率**: GPU 利用率，训练时间
- **泛化能力**: 训练 vs 验证 loss 差距

### 实验状态跟踪 (配置文件已更新 - 2025-01-23)

| 实验版本           | 状态        | 开始时间 | 完成时间 | 最佳验证 Loss | RSIEval 准确率 | 备注                     |
| ------------------ | ----------- | -------- | -------- | ------------- | -------------- | ------------------------ |
| Conservative       | ✅ 已更新   | -        | -        | -             | -              | batch_size: 24→12        |
| Grid V1            | ❌ 已标记   | -        | -        | -             | -              | 标记为 SKIPPED，batch→12 |
| Grid V2            | ✅ 已更新   | -        | -        | -             | -              | batch_size: 14→10        |
| Grid V3            | ❌ 已标记   | -        | -        | -             | -              | 标记为 SKIPPED，batch→12 |
| Grid V4            | ✅ 已更新   | -        | -        | -             | -              | batch_size: 12→8         |
| V5 (新)            | ✅ 已创建   | -        | -        | -             | -              | batch_size=6，最安全配置 |
| Ultra Conservative | ✅ 无需更新 | -        | -        | -             | -              | batch_size=12 已合适     |

### 🚨 OOM 经验总结 (重要!)

**发现**: Conservative 配置 (batch_size=24) 在 48GB GPU 上 OOM!

**错误信息**:

```
CUDA out of memory. Tried to allocate 472.00 MiB.
GPU 0 has a total capacity of 47.38 GiB of which 368.69 MiB is free.
Including non-PyTorch memory, this process has 47.01 GiB memory in use.
```

**关键洞察**:

1. **InstructBLIP 比预期更耗显存**: 47GB+ 被占用
2. **48GB 不等于可用 48GB**: 实际可用约 47.4GB
3. **LoRA 也有显存开销**: Q-Former LoRA 增加额外显存需求
4. **高分辨率图像**: RSICap 图像可能比预期占用更多显存

**新的 Batch Size 指导原则**:

- **batch_size ≤ 12**: 相对安全
- **batch_size ≤ 8**: 高 rank 时推荐
- **batch_size ≤ 6**: 最保守选择

### 预期结果分析

基于理论分析，预期各配置的表现：

1. **Conservative**: 最稳定，中等性能，推荐作为基线
2. **Grid V1**: 训练快速，可能性能良好，但需验证稳定性
3. **Grid V2**: 性能可能最佳，但训练时间较长
4. **Grid V3**: 极限测试，可能内存不足或不稳定
5. **Grid V4**: 理论性能最佳，但风险最高

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

## 📝 配置文件更新记录 (2025-01-23)

### 更新原因

基于 Conservative 配置 (batch_size=24) 在 48GB GPU 上 OOM 的经验，对所有 grid search 配置文件进行了安全性更新。

### 更新详情

| 配置文件                              | 原 batch_size | 新 batch_size | 状态变更       | 备注                     |
| ------------------------------------- | ------------- | ------------- | -------------- | ------------------------ |
| `conservative_optimal.yml`            | 16            | 12            | 已更新         | 进一步降低以确保安全     |
| `grid_search_v1_large_batch.yml`      | 16            | 12            | 标记为 SKIPPED | 原设计不可行，建议跳过   |
| `grid_search_v2_high_rank.yml`        | 14            | 10            | 已更新         | 高 rank 需要更小 batch   |
| `grid_search_v3_ultra_batch.yml`      | 16            | 12            | 标记为 SKIPPED | 原设计不可行，建议跳过   |
| `grid_search_v4_extreme_rank.yml`     | 12            | 8             | 已更新         | 极高 rank 需要最小 batch |
| `grid_search_v5_memory_optimized.yml` | -             | 6             | 新创建         | 最安全的配置选项         |
| `ultra_conservative.yml`              | 12            | 12            | 无需更新       | 已经是安全配置           |

### 新的安全指导原则

- **batch_size ≤ 12**: 相对安全的上限
- **batch_size ≤ 8**: 高 LoRA rank (r≥16) 时推荐
- **batch_size ≤ 6**: 最保守选择，适合极高 rank 或内存敏感场景

### 推荐执行顺序

1. **V5 (Memory Optimized)**: batch_size=6，最安全，优先测试
2. **Ultra Conservative**: batch_size=12，已验证可行
3. **Grid V2**: batch_size=10，高容量配置
4. **Grid V4**: batch_size=8，极限容量测试
5. **跳过 V1 和 V3**: 原设计不可行

---

## 🚀 新功能实现记录 (2025-01-24)

### 学习率调度器扩展

基于 Grid V4 过拟合问题的分析，实现了多种学习率调度策略：

#### 支持的调度器类型：

1. **Linear Scheduler** (原有)

   ```yaml
   scheduler_type: "linear"
   start_factor: 0.1
   warmup_steps: 100
   ```

2. **Cosine Annealing** (新增)

   ```yaml
   scheduler_type: "cosine"
   min_lr: 1e-6
   cosine_restarts: false
   ```

3. **Cosine with Warm Restarts** (新增)

   ```yaml
   scheduler_type: "cosine"
   cosine_restarts: true
   restart_period: 5
   restart_mult: 2
   ```

4. **Constant LR** (新增)
   ```yaml
   scheduler_type: "constant"
   ```

#### 实现文件：

- `module/training/scheduler_factory.py`: 调度器工厂类
- `module/config.py`: 新增配置选项

### 早停机制实现

为解决过拟合问题，实现了智能早停机制：

#### 配置参数：

```yaml
# 早停策略
early_stopping_enabled: true
early_stopping_patience: 3 # 验证loss不改善3个epoch就停止
min_delta: 0.001 # 最小改善阈值
```

#### 功能特性：

- ✅ 自动监控验证 loss 变化
- ✅ 可配置的耐心值和改善阈值
- ✅ 自动保存和恢复最佳模型权重
- ✅ 详细的早停日志输出

#### 实现文件：

- `module/training/early_stopping.py`: 早停机制类
- `module/training/trainer.py`: 集成到训练循环

### 测试用例

创建了完整的测试套件：

#### 测试文件：

- `tests/test_scheduler_early_stopping.py`: 单元测试
- `configs/test_cosine_early_stop.yml`: 测试配置

#### 测试覆盖：

- ✅ 早停机制的各种场景
- ✅ 所有调度器类型的创建
- ✅ 配置系统集成
- ✅ 错误处理

### 预期改进效果

基于 Grid V4 的过拟合分析，新功能预期带来：

1. **训练稳定性提升**：

   - 余弦调度器提供更平滑的学习率衰减
   - 早停防止过拟合，提升泛化能力

2. **训练效率提升**：

   - 自动早停减少不必要的训练时间
   - 智能学习率调度加速收敛

3. **模型性能提升**：
   - 预期验证 loss 从 1.399 改善至 1.25-1.30
   - 更好的训练/验证 loss 平衡

### 使用示例

```yaml
# 推荐的改进配置 (基于Grid V4优化)
name: "grid_v4_improved"
description: "Grid V4 with cosine scheduler and early stopping"

# 训练参数
num_epochs: 15 # 更多epoch，依赖早停
batch_size: 8
learning_rate: 0.0001 # 降低学习率
max_samples: 2068

# LoRA参数
lora_r: 32
lora_alpha: 64
lora_dropout: 0.1 # 增加dropout

# 新的调度器配置
scheduler_type: "cosine"
min_lr: 1e-6
warmup_steps: 250

# 早停配置
early_stopping_enabled: true
early_stopping_patience: 3
min_delta: 0.001

# 优化参数
max_grad_norm: 0.3
torch_dtype: "float32"
```

---

_最后更新: 2025-01-24_
_下次更新: 新功能的实验结果和性能对比_
