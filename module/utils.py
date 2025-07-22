"""
Utility functions for LoRA fine-tuning.
Agile approach: Basic utilities, can be expanded as needed.
"""

import os
import json
import torch
from PIL import Image


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'trainable_percent': (trainable_params / total_params) * 100
    }


def format_number(num):
    """Format large numbers for display"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def save_training_log(results, config, save_path):
    """Save training results and config to JSON"""
    log_data = {
        'config': config.__dict__,
        'results': results,
        'parameter_count': count_parameters(results.get('model', None)) if 'model' in results else None
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"Training log saved to {save_path}")


def load_image_safe(image_path, default_size=(224, 224)):
    """Safely load an image, return black image if fails"""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return Image.new('RGB', default_size, color='black')


def check_gpu_memory():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        free = total - reserved
        
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'utilization': (reserved / total) * 100
        }
    else:
        return None


def print_gpu_summary():
    """Print GPU memory summary"""
    memory_info = check_gpu_memory()
    if memory_info:
        print("GPU Memory Summary:")
        print(f"  Total: {memory_info['total']:.2f}GB")
        print(f"  Reserved: {memory_info['reserved']:.2f}GB ({memory_info['utilization']:.1f}%)")
        print(f"  Free: {memory_info['free']:.2f}GB")
    else:
        print("CUDA not available")


def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)
    return path


# Simple early stopping class for future use
class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop training