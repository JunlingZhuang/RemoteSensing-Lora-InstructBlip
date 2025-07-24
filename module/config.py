"""
Configuration management for LoRA fine-tuning.
Agile approach: Simple config class, expandable for future hyperparameter sweeps.
"""

class Config:
    """Simple configuration class for LoRA fine-tuning"""
    
    def __init__(self):
        # Model settings (based on successful inference from debug notebook)
        self.model_name = "Salesforce/instructblip-flan-t5-xl"
        self.device = "cuda"
        self.torch_dtype = "float32"
        
        # LoRA settings (conservative for MVP stability)
        self.lora_r = 8   # Lower rank for stability
        self.lora_alpha = 16  # Proportional scaling
        self.lora_dropout = 0.1
        # 只微调 Q-Former，不微调 Language Model (简化版本)
        self.target_modules = ["query", "key", "value", "dense"]
        # 注意：需要在 LoRA 应用时指定只应用到 qformer 模块
        
        # Training settings (ultra-conservative for MVP success)
        self.learning_rate = 1e-6  # Ultra-low to prevent Q-Former LoRA instability
        self.batch_size = 4  # Start small for 3090
        self.num_epochs = 3
        self.warmup_steps = 100
        self.weight_decay = 0.0
        self.max_grad_norm = 0.5  # Gradient clipping for stability

        # Learning Rate Scheduler settings
        self.scheduler_type = "linear"  # Options: "linear", "cosine", "constant"
        self.start_factor = 0.1  # For linear scheduler warmup
        self.min_lr = 1e-7  # Minimum learning rate for cosine scheduler
        self.cosine_restarts = False  # Whether to use cosine annealing with restarts

        # Early Stopping settings
        self.early_stopping_patience = 3  # 验证loss不改善3个epoch就停止
        self.min_delta = 0.001  # 最小改善阈值
        self.early_stopping_enabled = False  # 默认关闭，可在配置文件中启用
        
        # Data settings (80/20 split from plan)
        self.train_split = 0.8
        self.val_split = 0.2
        self.random_seed = 42
        
        # Data paths are set by individual dataset loaders with their own defaults
        
        # Output settings
        self.save_dir = "checkpoints"
        self.log_every = 10  # Log every N steps
        self.save_every_epoch = True
        
    def update(self, **kwargs):
        """Update config values - useful for hyperparameter sweeps later"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid config parameter")
    
    def __str__(self):
        """Pretty print config for logging"""
        items = []
        for key, value in self.__dict__.items():
            items.append(f"{key}: {value}")
        return "\n".join(items)


# Default config instance
default_config = Config()