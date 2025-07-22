"""
Simple LoRA trainer for MVP
"""

import os
import time
import torch
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


class SimpleLoRATrainer:
    """Simple LoRA trainer for InstructBLIP"""
    
    def __init__(self, base_model, train_loader, val_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup LoRA
        print("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=8,  # rank
            lora_alpha=32,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(base_model, lora_config)
        self.model.train()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
    def count_trainable_parameters(self):
        """Count trainable parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return f"{trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move to GPU
            pixel_values = batch['pixel_values'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move to GPU
                pixel_values = batch['pixel_values'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        
        # Also save LoRA weights separately
        lora_path = path.replace('.pth', '_lora.pth')
        self.model.save_pretrained(os.path.dirname(lora_path))
        print(f"LoRA weights saved to {os.path.dirname(lora_path)}")