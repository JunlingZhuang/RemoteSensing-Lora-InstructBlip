# 推理使用指南 / Inference Guide

## 概述

本推理模块支持三种模型类型的推理：
- **Native InstructBLIP**: 原生的InstructBLIP模型
- **Native BLIP-2**: 原生的BLIP-2模型  
- **LoRA Fine-tuned**: LoRA微调后的InstructBLIP模型

## 🚀 快速开始

### 1. 单张图片推理

```bash
# 使用原生InstructBLIP
python inference_demo.py -i image.jpg

# 使用LoRA微调模型
python inference_demo.py -i image.jpg --model lora --lora-path ./saved_models/best_model

# 使用BLIP-2
python inference_demo.py -i image.jpg --model blip2
```

### 2. 批量图片推理

```bash
# 批量处理文件夹中的所有图片
python inference_demo.py --batch ./test_images --model lora --lora-path ./saved_models/best_model -o results.json
```

### 3. 模型对比

```bash
# 对比所有可用模型
python inference_demo.py -i image.jpg --model all --lora-path ./saved_models/best_model --compare -o comparison.json
```

## 🔧 高级用法

### 自定义参数 (使用debug notebook中的成功配置)

```bash
# 自定义生成参数 (默认来自成功的debug_instructblip.ipynb配置)
python inference_demo.py -i image.jpg \
    --max-tokens 300 \
    --num-beams 1 \
    --temperature 1.0 \
    --top-p 0.9 \
    --repetition-penalty 1.0 \
    --instruction "详细描述这张遥感图像中的地物特征"
```

### Python API 使用

```python
from module.inference.inferencer import ModelInferencer, quick_inference

# 1. 快速推理 (使用默认的成功配置)
caption = quick_inference("image.jpg", model_type="instructblip")
print(caption)

# 2. 详细推理 - 自定义生成参数
inferencer = ModelInferencer(model_type="lora", model_path="./saved_models/best_model")

# 使用debug notebook中成功的配置
caption = inferencer.generate_caption(
    "image.jpg", 
    "描述这张图像",
    max_new_tokens=300,
    num_beams=1,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.0,
    temperature=1.0
)

# 3. 更新默认生成配置
inferencer.update_generation_config(
    max_new_tokens=200,
    temperature=0.8
)

# 4. 批量推理
results = inferencer.batch_inference(["img1.jpg", "img2.jpg"])

# 5. 模型对比
native_inferencer = ModelInferencer(model_type="instructblip")
lora_inferencer = ModelInferencer(model_type="lora", model_path="./saved_models/best_model")
comparison = native_inferencer.compare_models("image.jpg", lora_inferencer)
```

## 📊 模型对比示例

运行模型对比后，你会看到类似输出：

```
🖼️ Image: remote_sensing_image.jpg
📝 Instruction: Describe this remote sensing image in detail.

🤖 INSTRUCTBLIP:
   This image shows an aerial view of a residential area with buildings and roads.

🤖 BLIP2:
   a photo of buildings and roads from above

🤖 LORA:
   This remote sensing image captures a dense residential area with approximately 
   25 houses arranged in a grid pattern. The houses have red and brown rooftops, 
   surrounded by green vegetation. A main road runs through the center from north to south.
```

## 🔄 模型加载方式差异

### LoRA模型加载 (两步加载)
1. 首先加载base model (InstructBLIP)
2. 然后加载LoRA权重并合并

```python
# 内部实现
base_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model = PeftModel.from_pretrained(base_model, lora_path)
```

### 原生模型加载 (直接加载)
```python
# 直接加载
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
```

## 📝 输出格式

### 单张推理输出
```json
{
  "image_path": "image.jpg",
  "model_type": "lora",
  "instruction": "Describe this image",
  "caption": "Generated caption text",
  "config": {
    "max_new_tokens": 300,
    "temperature": 1.0
  }
}
```

### 批量推理输出
```json
{
  "batch_directory": "./test_images",
  "model_type": "lora", 
  "total_images": 10,
  "successful": 9,
  "results": [
    {
      "image_path": "img1.jpg",
      "caption": "Caption for image 1",
      "success": true
    },
    {
      "image_path": "img2.jpg", 
      "error": "File not found",
      "success": false
    }
  ]
}
```

### 模型对比输出
```json
{
  "image_path": "image.jpg",
  "instruction": "Describe this image",
  "models_compared": ["instructblip", "lora"],
  "results": {
    "instructblip": "Native model caption",
    "lora": "LoRA model caption"
  }
}
```

## ⚙️ 配置参数

**重要变更**: Generation settings现在从Config中分离，推理时可以灵活调整

### 默认生成参数 (来自成功的debug_instructblip.ipynb)

```python
# 这些是内置的默认值，基于你notebook中成功的配置
generation_config = {
    "max_new_tokens": 300,        # 最大生成token数
    "num_beams": 1,               # beam search数量 
    "do_sample": True,            # 是否采样
    "top_p": 0.9,                 # top-p采样
    "temperature": 1.0,           # 生成温度
    "repetition_penalty": 1.0     # 重复惩罚
}
```

### 推理时自定义参数

```python
# 方法1: 在generate_caption时传递参数
caption = inferencer.generate_caption(
    "image.jpg", 
    instruction="描述图像",
    max_new_tokens=200,           # 覆盖默认值
    temperature=0.8               # 覆盖默认值
)

# 方法2: 更新默认配置
inferencer.update_generation_config(
    max_new_tokens=200,
    temperature=0.8
)
```

### Config类现在只包含

- 模型设置 (model_name, device等)
- LoRA设置 (lora_r, lora_alpha等) 
- 训练设置 (learning_rate, batch_size等)
- 数据设置 (数据路径, split等)

## 🧪 测试推理功能

```bash
# 运行推理模块测试
python module/tests/test_inference.py

# 或使用测试运行器
python module/tests/run_tests.py --pattern inference
```

## 🔍 故障排除

### 常见问题

1. **LoRA模型加载失败**
   - 检查模型路径是否正确
   - 确保LoRA权重文件存在

2. **显存不足**
   - 减少batch_size
   - 使用CPU推理：设置device="cpu"

3. **模型下载失败**
   - 检查网络连接
   - 使用镜像源或本地模型

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查GPU内存
from module.utils import print_gpu_summary
print_gpu_summary()
```

## 📈 性能优化

1. **使用GPU加速**: 确保CUDA可用
2. **批量处理**: 对多张图片使用batch_inference
3. **模型缓存**: 避免重复加载同一模型
4. **精度优化**: 使用float16精度节省显存

## 🎯 应用场景

- **遥感图像分析**: 自动生成遥感图像描述
- **模型评估**: 对比不同模型的性能
- **数据标注**: 批量生成图像标注
- **研究实验**: 评估微调效果