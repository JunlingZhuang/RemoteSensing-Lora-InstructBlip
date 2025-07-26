# LoRA InstructBLIP Hyperparameter Tuning Notes

Just documenting my journey fine-tuning InstructBLIP with LoRA on remote sensing data. Lots of trial and error here.

## The Setup

- **Model**: InstructBLIP + LoRA adapters
- **Dataset**: RSICap (remote sensing image captions)
- **Hardware**: 48GB GPU (thought this would be enough... spoiler: it wasn't always)

## Early Experiments (The Painful Learning Phase)

### Experiment 1: The NaN Disaster

Started with what seemed like reasonable settings:

```python
config = {
    "batch_size": 2,
    "learning_rate": 5e-6,
    "lora_r": 4,
    "lora_alpha": 8,
    "torch_dtype": "float16"  # This was the mistake
}
```

**What happened**: First batch worked fine, then everything exploded into NaN values. Took me way too long to figure out that float16 precision wasn't enough for LoRA fine-tuning.

**Fix**: Switch to float32. Problem solved.

### Experiment 2: Getting Stable Training

```python
config = {
    "batch_size": 8,
    "learning_rate": 5e-6,
    "lora_r": 8,
    "lora_alpha": 16,
    "torch_dtype": "float32"
}
```

Finally got stable training! Loss went from 2.06 to 1.55 over 10 epochs. Not spectacular, but at least it worked.

### Experiment 3: Being Too Aggressive

Got impatient and cranked everything up:

```python
config = {
    "learning_rate": 1e-3,  # Way too high
    "lora_r": 16,           # Too much capacity
    "lora_alpha": 32
}
```

Classic mistake. First few epochs looked amazing (1.89 → 1.71 → 1.69), then everything went to hell. Validation loss started climbing at epoch 3. Textbook overfitting.

## The 48GB GPU Reality Check

Thought having 48GB would mean I could use huge batch sizes. Wrong.

Tried batch_size=24 and got:
```
CUDA out of memory. Tried to allocate 472.00 MiB.
GPU 0 has a total capacity of 47.38 GiB of which 368.69 MiB is free.
```

Turns out InstructBLIP is a memory hog. Had to scale back to batch_size=12 or lower.

## Grid Search Results (The Good Stuff)

After all the trial and error, ran a proper grid search. Here are the configs that actually worked:

### V6 Experiments (The Winners)

| Experiment | LoRA r | Alpha | Scheduler | Dropout | Best Val Loss |
|------------|--------|-------|-----------|---------|---------------|
| exp3       | 24     | 64    | Linear    | 0.10    | **1.2727**    |
| exp5       | 32     | 64    | Linear    | 0.05    | 1.2738        |
| exp8       | 32     | 48    | Linear    | 0.05    | 1.2876        |

**Key findings**:
- Linear scheduler consistently beats cosine
- r=24, alpha=64 is the sweet spot
- Higher dropout (0.10) helps with lower ranks

## What Actually Matters

After running 14+ different configurations, here's what I learned:

### 1. Numerical Stability First
- Always use float32 for LoRA fine-tuning
- Gradient clipping is essential (max_grad_norm=1.0)
- Custom forward pass needed due to HF/LoRA conflicts

### 2. Learning Rate Scheduler Choice
Linear scheduling works way better than cosine for this task. Cosine annealing seems to decay too aggressively early on.

### 3. LoRA Configuration
- **r=24**: Best balance of capacity and stability
- **alpha=64**: Works well with r=24 (ratio of ~2.67)
- **dropout=0.10**: More regularization needed for lower ranks

### 4. Batch Size Reality
- batch_size ≤ 12 is safe on 48GB
- batch_size ≤ 8 for high LoRA ranks
- batch_size ≤ 6 for maximum safety

## Current Best Config

```python
BEST_CONFIG = {
    "learning_rate": 2e-4,
    "lora_r": 24,
    "lora_alpha": 64,
    "lora_dropout": 0.10,
    "batch_size": 8,
    "scheduler_type": "linear",
    "torch_dtype": "float32",
    "max_grad_norm": 1.0,
    "warmup_steps": 100
}
```

This gets validation loss down to 1.2727, which is pretty solid.
