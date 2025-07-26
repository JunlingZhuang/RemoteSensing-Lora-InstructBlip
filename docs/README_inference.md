# Inference Guide

## Overview

This inference module supports three types of model inference:

- **Native InstructBLIP**: Original InstructBLIP model
- **Native BLIP-2**: Original BLIP-2 model
- **LoRA Fine-tuned**: LoRA fine-tuned InstructBLIP model

## Quick Start

### 1. Single Image Inference

```bash
# Use native InstructBLIP
python inference_demo.py -i image.jpg

# Use LoRA fine-tuned model
python inference_demo.py -i image.jpg --model lora --lora-path ./saved_models/best_model

# Use BLIP-2
python inference_demo.py -i image.jpg --model blip2
```

### 2. Batch Image Inference

```bash
# Batch process all images in folder
python inference_demo.py --batch ./test_images --model lora --lora-path ./saved_models/best_model -o results.json
```

### 3. Model Comparison

```bash
# Compare all available models
python inference_demo.py -i image.jpg --model all --lora-path ./saved_models/best_model --compare -o comparison.json
```

## Advanced Usage

### Custom Parameters (using successful configurations from debug notebook)

```bash
# Custom generation parameters (defaults from successful debug_instructblip.ipynb configuration)
python inference_demo.py -i image.jpg \
    --max-tokens 300 \
    --num-beams 1 \
    --temperature 1.0 \
    --top-p 0.9 \
    --repetition-penalty 1.0 \
    --instruction "Describe the land features in this remote sensing image in detail"
```

### Python API Usage

```python
from module.inference.inferencer import ModelInferencer, quick_inference

# 1. Quick inference (using default successful configuration)
caption = quick_inference("image.jpg", model_type="instructblip")
print(caption)

# 2. Detailed inference - custom generation parameters
inferencer = ModelInferencer(model_type="lora", model_path="./saved_models/best_model")

# Use successful configuration from debug notebook
caption = inferencer.generate_caption(
    "image.jpg",
    "Describe this image",
    max_new_tokens=300,
    num_beams=1,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.0,
    temperature=1.0
)

# 3. Update default generation configuration
inferencer.update_generation_config(
    max_new_tokens=200,
    temperature=0.8
)

# 4. Batch inference
results = inferencer.batch_inference(["img1.jpg", "img2.jpg"])

# 5. Model comparison
native_inferencer = ModelInferencer(model_type="instructblip")
lora_inferencer = ModelInferencer(model_type="lora", model_path="./saved_models/best_model")
comparison = native_inferencer.compare_models("image.jpg", lora_inferencer)
```

## Model Comparison Example

After running model comparison, you will see output similar to:

```
Image: remote_sensing_image.jpg
Instruction: Describe this remote sensing image in detail.

INSTRUCTBLIP:
   This image shows an aerial view of a residential area with buildings and roads.

BLIP2:
   a photo of buildings and roads from above

LORA:
   This remote sensing image captures a dense residential area with approximately
   25 houses arranged in a grid pattern. The houses have red and brown rooftops,
   surrounded by green vegetation. A main road runs through the center from north to south.
```

## Model Loading Method Differences

### LoRA Model Loading (Two-step loading)

1. First load base model (InstructBLIP)
2. Then load LoRA weights and merge

```python
# Internal implementation
base_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model = PeftModel.from_pretrained(base_model, lora_path)
```

### Native Model Loading (Direct loading)

```python
# Direct loading
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
```

## Output Format

### Single Image Inference Output

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

### Batch Inference Output

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

### Model Comparison Output

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

## Configuration Parameters

**Important Change**: Generation settings are now separated from Config, allowing flexible adjustment during inference

### Default Generation Parameters (from successful debug_instructblip.ipynb)

```python
# These are built-in defaults based on successful configurations from your notebook
generation_config = {
    "max_new_tokens": 300,        # Maximum number of tokens to generate
    "num_beams": 1,               # Number of beams for beam search
    "do_sample": True,            # Whether to use sampling
    "top_p": 0.9,                 # Top-p sampling
    "temperature": 1.0,           # Generation temperature
    "repetition_penalty": 1.0     # Repetition penalty
}
```

### Custom Parameters During Inference

```python
# Method 1: Pass parameters in generate_caption
caption = inferencer.generate_caption(
    "image.jpg",
    instruction="Describe the image",
    max_new_tokens=200,           # Override default value
    temperature=0.8               # Override default value
)

# Method 2: Update default configuration
inferencer.update_generation_config(
    max_new_tokens=200,
    temperature=0.8
)
```

### Config Class Now Only Contains

- Model settings (model_name, device, etc.)
- LoRA settings (lora_r, lora_alpha, etc.)
- Training settings (learning_rate, batch_size, etc.)
- Data settings (data paths, split, etc.)

## Testing Inference Functionality

```bash
# Run inference module tests
python module/tests/test_inference.py

# Or use test runner
python module/tests/run_tests.py --pattern inference
```

## Troubleshooting

### Common Issues

1. **LoRA Model Loading Failed**

   - Check if model path is correct
   - Ensure LoRA weight files exist

2. **Out of Memory**

   - Reduce batch_size
   - Use CPU inference: set device="cpu"

3. **Model Download Failed**
   - Check network connection
   - Use mirror sources or local models

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU memory
from module.utils import print_gpu_summary
print_gpu_summary()
```

## Performance Optimization

1. **Use GPU Acceleration**: Ensure CUDA is available
2. **Batch Processing**: Use batch_inference for multiple images
3. **Model Caching**: Avoid repeatedly loading the same model
4. **Precision Optimization**: Use float16 precision to save memory

## Application Scenarios

- **Remote Sensing Image Analysis**: Automatically generate remote sensing image descriptions
- **Model Evaluation**: Compare performance of different models
- **Data Annotation**: Batch generate image annotations
- **Research Experiments**: Evaluate fine-tuning effects
