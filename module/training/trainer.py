"""
Simple trainer for LoRA fine-tuning with loss monitoring.
Agile approach: Basic training loop that shows train/val loss, can be enhanced later.
"""

import os
import sys
import time
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import new modules
from .early_stopping import EarlyStopping
from .scheduler_factory import create_scheduler, print_scheduler_info, get_current_lr


class LoRATrainer:
    """Simple trainer for LoRA fine-tuning"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer (following LoRA best practices)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler using factory
        self.scheduler, self.scheduler_info = create_scheduler(
            self.optimizer, config, train_loader
        )
        print_scheduler_info(self.scheduler_info)
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.learning_rates = []  # Track learning rate changes
        self.epoch_details = []   # Detailed epoch information

        # Early stopping setup
        self.early_stopping = None
        if getattr(config, 'early_stopping_enabled', False):
            self.early_stopping = EarlyStopping(
                patience=getattr(config, 'early_stopping_patience', 3),
                min_delta=getattr(config, 'min_delta', 0.001),
                restore_best_weights=True,
                verbose=True
            )
            print(f"🛑 Early stopping enabled: patience={self.early_stopping.patience}, "
                  f"min_delta={self.early_stopping.min_delta}")

        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        epoch_start = time.time()

        # Create simple progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )

        for step, batch in enumerate(pbar):
            # Forward pass
            outputs = self.model.forward(batch)

            # Check if forward returned None (NaN detected)
            if outputs is None:
                pbar.set_postfix({"Status": "SKIP", "Reason": "NaN"})
                continue

            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):
                pbar.set_postfix({"Status": "SKIP", "Reason": "Loss-NaN"})
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if hasattr(self.config, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            current_avg_loss = total_loss / num_batches
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg": f"{current_avg_loss:.4f}",
                "LR": f"{current_lr:.1e}"
            })
        
        # Close progress bar
        pbar.close()

        if num_batches > 0:
            avg_train_loss = total_loss / num_batches
        else:
            avg_train_loss = float('inf')
            print("WARNING: No valid batches processed!")

        epoch_time = time.time() - epoch_start

        # Store training history
        self.train_losses.append(avg_train_loss)
        self.epoch_times.append(epoch_time)

        # Record current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"📊 Average training loss: {avg_train_loss:.4f}")
        print(f"🔢 Processed {num_batches}/{len(self.train_loader)} valid batches")

        return avg_train_loss, epoch_time
    
    def validate(self, epoch):
        """Validate the model with progress bar"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Create simple validation progress bar
        val_pbar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False
        )

        with torch.no_grad():
            for batch in val_pbar:
                outputs = self.model.forward(batch)

                # Check if forward returned None
                if outputs is None:
                    val_pbar.set_postfix({"Status": "SKIP-None"})
                    continue

                loss = outputs.loss

                # Check for NaN loss
                if torch.isnan(loss):
                    val_pbar.set_postfix({"Status": "SKIP-NaN"})
                    continue

                total_loss += loss.item()
                num_batches += 1

                # Update validation progress
                current_avg_loss = total_loss / num_batches
                val_pbar.set_postfix({"Val Loss": f"{current_avg_loss:.4f}"})

        # Close validation progress bar
        val_pbar.close()

        if num_batches > 0:
            avg_val_loss = total_loss / num_batches
        else:
            avg_val_loss = float('inf')
            print("WARNING: No valid validation batches processed!")

        # Store validation history
        self.val_losses.append(avg_val_loss)

        # Record detailed epoch information
        epoch_info = {
            "epoch": len(self.train_losses),
            "train_loss": self.train_losses[-1],
            "val_loss": avg_val_loss,
            "learning_rate": self.learning_rates[-1] if self.learning_rates else 0,
            "epoch_time": self.epoch_times[-1] if self.epoch_times else 0,
            "train_batches": len(self.train_loader),
            "val_batches": len(self.val_loader)
        }
        self.epoch_details.append(epoch_info)

        print(f"📈 Validation loss: {avg_val_loss:.4f}")
        print(f"🔢 Processed {num_batches}/{len(self.val_loader)} valid validation batches")

        return avg_val_loss
    
    def train(self):
        """Full training loop with loss monitoring"""
        print("Starting LoRA fine-tuning...")
        print(f"Config:\n{self.config}")
        print(f"\nInitial memory usage: {self.model.get_memory_usage()}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.epoch_times.append(epoch_time)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.config.save_dir, "best_model")
                self.model.save_pretrained(best_model_path)
                print(f"New best model saved! Val loss: {val_loss:.4f}")

            # Check early stopping
            if self.early_stopping is not None:
                should_stop = self.early_stopping(val_loss, self.model.model, epoch + 1)
                if should_stop:
                    print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                    # Restore best model weights
                    self.early_stopping.restore_best_model(self.model.model)
                    break

            # Save checkpoint every epoch if configured
            if self.config.save_every_epoch:
                checkpoint_path = os.path.join(self.config.save_dir, f"epoch_{epoch + 1}")
                self.model.save_pretrained(checkpoint_path)

            print(f"Epoch {epoch + 1} summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Memory: {self.model.get_memory_usage()}")
            if self.early_stopping is not None:
                print(f"  Early Stop Counter: {self.early_stopping.counter}/{self.early_stopping.patience}")
        
        # Final summary
        self.print_summary()
        self.plot_losses()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'total_time': sum(self.epoch_times)
        }
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"Total time: {sum(self.epoch_times):.2f}s")
        print(f"Average time per epoch: {sum(self.epoch_times)/len(self.epoch_times):.2f}s")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val loss: {self.val_losses[-1]:.4f}")
        print(f"Best val loss: {min(self.val_losses):.4f}")
        print(f"Final memory usage: {self.model.get_memory_usage()}")
    
    def plot_losses(self):
        """Plot training and validation losses with learning rate"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        epochs = range(1, len(self.train_losses) + 1)

        # Plot 1: Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', marker='o', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)

        # Mark best validation loss
        best_epoch = self.val_losses.index(min(self.val_losses)) + 1
        best_val_loss = min(self.val_losses)
        ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
        ax1.annotate(f'Best: {best_val_loss:.4f}',
                     xy=(best_epoch, best_val_loss),
                     xytext=(10, 10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        if self.learning_rates:
            ax2.plot(epochs, self.learning_rates, 'g-', label='Learning Rate', marker='d', linewidth=2)
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to {plot_path}")

        plt.close()  # Don't show, just save

    def save_checkpoint(self, save_path, epoch, train_loss, val_loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epoch_details': self.epoch_details,
            'config': self.config.__dict__
        }

        # Save LoRA model
        self.model.save_pretrained(save_path)

        # Save checkpoint info
        checkpoint_file = os.path.join(save_path, 'checkpoint.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"Checkpoint saved to {save_path}")

    def save_training_summary(self):
        """Save comprehensive training summary at the end"""
        if not self.train_losses:
            return

        # Create detailed summary
        summary = {
            "training_completed": True,
            "total_epochs": len(self.train_losses),
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_loss": min(self.val_losses) if self.val_losses else None,
            "best_val_epoch": self.val_losses.index(min(self.val_losses)) + 1 if self.val_losses else None,
            "total_training_time": sum(self.epoch_times),
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            "epoch_details": self.epoch_details,
            "config": self.config.__dict__
        }

        # Save summary
        summary_path = os.path.join(self.config.save_dir, "training_summary_detailed.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Detailed training summary saved to {summary_path}")

        # Generate plots
        self.plot_losses()

        return summary_path