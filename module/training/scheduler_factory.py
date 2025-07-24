"""
Learning Rate Scheduler Factory for LoRA training.
Supports multiple scheduler types: Linear, Cosine, Constant.
"""

import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR, CosineAnnealingWarmRestarts


def create_scheduler(optimizer, config, train_loader):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object with scheduler settings
        train_loader: Training data loader (for calculating total steps)
        
    Returns:
        scheduler: PyTorch learning rate scheduler
        scheduler_info: Dictionary with scheduler information
    """
    scheduler_type = getattr(config, 'scheduler_type', 'linear').lower()
    total_steps = len(train_loader) * config.num_epochs
    
    scheduler_info = {
        "type": scheduler_type,
        "total_steps": total_steps,
        "warmup_steps": config.warmup_steps
    }
    
    if scheduler_type == "linear":
        # Linear warmup + constant learning rate
        start_factor = getattr(config, 'start_factor', 0.1)
        scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=config.warmup_steps
        )
        scheduler_info.update({
            "start_factor": start_factor,
            "description": f"Linear warmup from {start_factor} to 1.0 over {config.warmup_steps} steps"
        })
        
    elif scheduler_type == "cosine":
        # Cosine annealing with optional warm restarts
        min_lr = getattr(config, 'min_lr', 1e-7)
        cosine_restarts = getattr(config, 'cosine_restarts', False)
        
        if cosine_restarts:
            # Cosine annealing with warm restarts
            T_0 = getattr(config, 'restart_period', config.num_epochs // 3)  # Initial restart period
            T_mult = getattr(config, 'restart_mult', 2)  # Restart period multiplier
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=min_lr
            )
            scheduler_info.update({
                "min_lr": min_lr,
                "restart_period": T_0,
                "restart_mult": T_mult,
                "description": f"Cosine annealing with warm restarts (T_0={T_0}, T_mult={T_mult}, min_lr={min_lr})"
            })
        else:
            # Standard cosine annealing
            # Calculate effective epochs after warmup
            effective_epochs = max(1, config.num_epochs - (config.warmup_steps // len(train_loader)))
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=effective_epochs,
                eta_min=min_lr
            )
            scheduler_info.update({
                "min_lr": min_lr,
                "T_max": effective_epochs,
                "description": f"Cosine annealing over {effective_epochs} epochs (min_lr={min_lr})"
            })
            
    elif scheduler_type == "constant":
        # Constant learning rate (no scheduling)
        scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=1  # Minimal constant scheduler
        )
        scheduler_info.update({
            "description": "Constant learning rate (no scheduling)"
        })
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: 'linear', 'cosine', 'constant'")
    
    return scheduler, scheduler_info


def create_warmup_cosine_scheduler(optimizer, config, train_loader):
    """
    Create a combined warmup + cosine annealing scheduler.
    This is a more sophisticated scheduler that combines linear warmup with cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object
        train_loader: Training data loader
        
    Returns:
        scheduler: Custom scheduler function
        scheduler_info: Dictionary with scheduler information
    """
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = config.warmup_steps
    min_lr = getattr(config, 'min_lr', 1e-7)
    base_lr = config.learning_rate
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    scheduler_info = {
        "type": "warmup_cosine",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "base_lr": base_lr,
        "description": f"Linear warmup ({warmup_steps} steps) + Cosine annealing (min_lr={min_lr})"
    }
    
    return scheduler, scheduler_info


def get_current_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def print_scheduler_info(scheduler_info):
    """Print scheduler configuration information."""
    print("📊 Learning Rate Scheduler Configuration:")
    print(f"   Type: {scheduler_info['type']}")
    print(f"   Description: {scheduler_info['description']}")
    print(f"   Total steps: {scheduler_info['total_steps']}")
    if 'warmup_steps' in scheduler_info:
        print(f"   Warmup steps: {scheduler_info['warmup_steps']}")
    print()
