#!/usr/bin/env python3
"""
Clean LoRA Training Script for InstructBLIP on RSICap
Supports YAML configuration files for flexible training
"""

import sys
import os
import json
import yaml
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'module'))

from config import Config
from data.rsicap_dataset import load_rsicap_data, RSICapDataset, collate_fn
from torch.utils.data import DataLoader
from models.lora_model import LoRAInstructBLIP
from training.trainer import LoRATrainer
import torch
import gc

# 强制清理
torch.cuda.empty_cache()
gc.collect()

# 中国镜像支持
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================================
# CONFIGURATION FILES TO RUN
# ============================================================================
# Define which config files to run (can be single or multiple)
CONFIG_FILES = [
    "configs/baseline_lora_instructblip.yml",
    # "configs/high_lr_experiment.yml",  # Add more configs here
    # "configs/large_batch_experiment.yml",
]

# ============================================================================
# YAML CONFIG LOADER
# ============================================================================
def load_yaml_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    print(f"📋 Loaded config: {yaml_config.get('name', 'unnamed')}")
    if 'description' in yaml_config:
        print(f"📝 Description: {yaml_config['description']}")

    return yaml_config

def train_single_config(config_path):
    """Train with a single YAML configuration"""

    # Load YAML configuration
    yaml_config = load_yaml_config(config_path)

    print("="*60)
    print("🚀 LoRA Training - InstructBLIP on RSICap")
    print(f"📋 Config: {yaml_config.get('name', 'unnamed')}")
    print("="*60)

    # Create config object
    config = Config()

    # Apply YAML config (skip metadata fields)
    metadata_fields = {'name', 'description'}
    for key, value in yaml_config.items():
        if key not in metadata_fields:
            setattr(config, key, value)

    # Create output directory using config name
    config_name = yaml_config.get('name', 'unnamed_config')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"checkpoints/{config_name}_{timestamp}"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 数据路径已在 config.py 中正确设置

    # 显示配置
    print(f"📋 Training Configuration:")
    print(f"  LoRA rank (r): {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max samples: {config.max_samples}")
    print(f"  Save dir: {config.save_dir}")
    print()

    # 加载组件
    print("Loading LoRA model...")
    model = LoRAInstructBLIP(config)
    print("✅ Model loaded successfully!")

    print("Loading dataset...")
    train_loader, val_loader, processor = load_rsicap_data(config)
    print("✅ Dataset loaded successfully!")

    # 限制数据量
    if config.max_samples:
        print(f"🔢 Limiting training to {config.max_samples} samples")
        limited_train_data = list(train_loader.dataset.data)[:config.max_samples]
        limited_train_dataset = RSICapDataset(
            limited_train_data,
            train_loader.dataset.images_dir,
            train_loader.dataset.processor,
            'train'
        )
        train_loader = DataLoader(
            limited_train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print(f"✅ Limited to {len(train_loader)} batches per epoch")

    # 创建训练器
    print("Creating trainer...")
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    print("✅ Trainer created successfully!")

    # 验证 LoRA 配置
    model.verify_lora_training()

    # 开始训练
    print("\n🚀 Starting LoRA training...")
    print("="*60)

    latest_checkpoint_path = None

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")

        # 训练
        train_loss, epoch_time = trainer.train_epoch(epoch)

        # 验证
        val_loss = trainer.validate(epoch)

        print(f"Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Time={epoch_time:.1f}s")

        # 保存每个epoch的历史记录
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            "learning_rate": trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else None,
            "timestamp": datetime.now().isoformat()
        }

        # 保存单个epoch总结
        epoch_file = os.path.join(config.save_dir, f"epoch_{epoch+1}_summary.json")
        with open(epoch_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_summary, f, ensure_ascii=False, indent=2)

        # 保存累积历史
        history_file = os.path.join(config.save_dir, "training_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {"epochs": []}

        history["epochs"].append(epoch_summary)
        history["config"] = yaml_config
        history["last_updated"] = datetime.now().isoformat()

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"📊 Epoch {epoch + 1} history saved")

        # 保存检查点
        if (epoch + 1) % 2 == 0 or epoch == config.num_epochs - 1:
            latest_checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            trainer.save_checkpoint(latest_checkpoint_path, epoch, train_loss, val_loss)
            print(f"💾 Checkpoint saved: {latest_checkpoint_path}")

    # 保存训练总结和生成可视化
    print("\n" + "="*60)
    print("🎉 Training Completed!")
    print("="*60)

    # 调用trainer的详细保存功能（包含可视化）
    if hasattr(trainer, 'save_training_summary'):
        detailed_summary_path = trainer.save_training_summary()
        print(f"📊 Detailed training summary with plots saved: {detailed_summary_path}")
    else:
        # 如果trainer没有这个方法，调用plot_losses
        if hasattr(trainer, 'plot_losses'):
            trainer.plot_losses()
            print(f"📈 Training curves saved to: {config.save_dir}/training_curves.png")

    # 保存基本总结
    summary = {
        "training_completed": datetime.now().isoformat(),
        "config": yaml_config,
        "final_losses": {
            "train_loss": trainer.train_losses[-1] if trainer.train_losses else None,
            "val_loss": trainer.val_losses[-1] if trainer.val_losses else None
        },
        "best_val_loss": min(trainer.val_losses) if trainer.val_losses else None,
        "total_epochs": len(trainer.train_losses),
        "total_training_time": sum(trainer.epoch_times) if hasattr(trainer, 'epoch_times') else None,
        "checkpoints_dir": config.save_dir
    }

    summary_path = os.path.join(config.save_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"📄 Basic training summary saved: {summary_path}")
    print(f"💾 Latest checkpoint: {latest_checkpoint_path}")
    print(f"📁 All files saved to: {config.save_dir}")
    print("\n✅ All done!")
    return latest_checkpoint_path

def main():
    """Main function to run training with multiple configurations"""

    print("🚀 Starting LoRA Training Pipeline")
    print(f"📁 Found {len(CONFIG_FILES)} configuration(s) to run")
    print("="*60)

    results = []

    for i, config_path in enumerate(CONFIG_FILES, 1):
        print(f"\n🔄 Running configuration {i}/{len(CONFIG_FILES)}: {config_path}")

        try:
            checkpoint_path = train_single_config(config_path)
            results.append({
                "config_path": config_path,
                "status": "success",
                "checkpoint": checkpoint_path
            })
            print(f"✅ Configuration {i} completed successfully!")

        except Exception as e:
            print(f"❌ Configuration {i} failed: {e}")
            results.append({
                "config_path": config_path,
                "status": "failed",
                "error": str(e)
            })

    # Print final summary
    print("\n" + "="*60)
    print("🎉 Training Pipeline Completed!")
    print("="*60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        print("\n📊 Successful Runs:")
        for result in successful:
            config_name = os.path.basename(result["config_path"])
            print(f"  - {config_name} → {result['checkpoint']}")

    if failed:
        print("\n💥 Failed Runs:")
        for result in failed:
            config_name = os.path.basename(result["config_path"])
            print(f"  - {config_name}: {result['error']}")

if __name__ == "__main__":
    main()
