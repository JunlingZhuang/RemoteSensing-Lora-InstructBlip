# LoRA Fine-tuning Module

Agile implementation of LoRA fine-tuning for InstructBLIP on RSICap dataset.

## Quick Start

1. **Check your data paths in `config.py`**:
   ```python
   rsicap_images_dir = "data/rsgpt_dataset/RSICap/images"
   rsicap_captions_file = "data/rsgpt_dataset/RSICap/captions.json"
   ```

2. **Test data loading**:
   ```bash
   python train_lora.py --test-data
   ```

3. **Test model loading**:
   ```bash
   python train_lora.py --test-model
   ```

4. **Start training**:
   ```bash
   python train_lora.py
   ```

## What You'll See

The training will show:
- Real-time loss curves (train & validation)
- Memory usage monitoring
- Progress logging every 10 steps
- Automatic model saving (best + every epoch)

## Expected Output

```
Starting LoRA fine-tuning...
Using GPU: NVIDIA GeForce RTX 3090
Data loaded successfully!
Train batches: 517, Val batches: 130

Epoch 1/3
--------------------------------------------------
Step 0/517: Loss: 2.3456, LR: 3.00e-06, Memory: Allocated: 10.1GB
...
Validation loss: 1.8234
New best model saved! Val loss: 1.8234

Final train loss: 1.2345
Best val loss: 1.6789
```

## Files Structure

```
module/
├── config.py              # Configuration
├── data/
│   └── rsicap_dataset.py  # Dataset loader
├── models/
│   └── lora_model.py      # LoRA wrapper
├── training/
│   └── trainer.py         # Training loop
└── utils.py               # Utilities
```

## Expanding for Hyperparameter Sweeps

The code is designed to easily support the full hyperparameter sweep from the plan:

```python
# Future: Multiple configs
configs = [
    Config().update(lora_r=8, lora_alpha=16),
    Config().update(lora_r=16, lora_alpha=32),
    Config().update(lora_r=32, lora_alpha=64),
]

for config in configs:
    trainer = LoRATrainer(model, train_loader, val_loader, config)
    results = trainer.train()
```

## Memory Requirements

- RTX 3090 (24GB): ✅ Should work fine
- RTX 3080 (10GB): Reduce batch_size to 2
- RTX 4060 (8GB): Reduce batch_size to 1

## Troubleshooting

1. **Out of memory**: Reduce `batch_size` in config.py
2. **Data not found**: Check paths in config.py
3. **Model loading fails**: Check internet connection for model download

## Next Steps

After getting basic training working:
1. Add data augmentation
2. Implement hyperparameter sweeps
3. Add RSIEval evaluation
4. Add baseline model comparisons