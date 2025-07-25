# Grid Search V6 Augmented Experiments

## 概述
这些配置文件是原始Grid Search V6实验的数据增强版本，使用了增强后的RSICap数据集。

## 数据集信息
- **原始数据集**: 2,068 samples
- **增强数据集**: 7,755 samples  
- **增强倍数**: 3.75x
- **数据源**: `data/rsgpt_dataset/RSICap_augmented`

## 实验配置对比

| 实验 | LoRA配置 | 学习率组 | 调度器 | Batch Size | 原始max_samples | 增强max_samples |
|------|----------|----------|--------|------------|----------------|----------------|
| Exp1 | r=16, α=64, d=0.05 | A (0.0001) | Cosine | 12 | 2,068 | 7,755 |
| Exp2 | r=16, α=48, d=0.1  | A (0.0001) | Linear | 12 | 2,068 | 7,755 |
| Exp3 | r=24, α=64, d=0.1  | A (0.0001) | Linear | 8  | 2,068 | 7,755 |
| Exp4 | r=24, α=48, d=0.05 | A (0.0001) | Cosine | 8  | 2,068 | 7,755 |
| Exp5 | r=32, α=64, d=0.05 | A (0.0001) | Linear | 8  | 2,068 | 7,755 |
| Exp6 | r=32, α=32, d=0.1  | B (0.0002) | Cosine | 8  | 2,068 | 7,755 |
| Exp7 | r=16, α=48, d=0.1  | B (0.0002) | Cosine | 8  | 2,068 | 7,755 |
| Exp8 | r=32, α=48, d=0.05 | A (0.0001) | Linear | 8  | 2,068 | 7,755 |

## 主要变化

### 1. 数据集大小
- 所有实验的 `max_samples` 从 2,068 增加到 7,755
- 使用完整的增强数据集进行训练

### 2. 数据源标识
- 添加了 `data_source: "augmented"` 标识
- 便于区分使用的数据集类型

### 3. 配置文件命名
- 原始: `grid_search_v6_improved_exp{N}.yml`
- 增强: `grid_search_v6_improved_exp{N}_augmented.yml`

## 预期效果

### 优势
1. **更多训练数据**: 3.75倍的数据量可能提高模型泛化能力
2. **数据多样性**: 数据增强引入的变化可能提高模型鲁棒性
3. **更好的收敛**: 更多数据可能帮助模型找到更好的局部最优解

### 注意事项
1. **训练时间**: 数据量增加3.75倍，训练时间会显著增长
2. **内存使用**: 需要确保有足够的内存处理更大的数据集
3. **过拟合风险**: 虽然数据更多，但仍需监控验证损失

## 使用方法

```bash
# 运行单个增强实验
python train_lora.py --config configs/grid_search_v6_improved_exp1_augmented.yml

# 批量运行所有增强实验
for i in {1..8}; do
    python train_lora.py --config configs/grid_search_v6_improved_exp${i}_augmented.yml
done
```

## 结果比较
建议将增强版本的结果与原始版本进行对比，分析数据增强对不同LoRA配置的影响。
