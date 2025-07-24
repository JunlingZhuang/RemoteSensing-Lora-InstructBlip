"""
Early Stopping mechanism for LoRA training.
Monitors validation loss and stops training when no improvement is observed.
"""

import numpy as np


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Internal state
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model=None, epoch=None):
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Current validation loss
            model: Model object (for saving best weights)
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should be stopped
        """
        # Check if this is an improvement
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch if epoch is not None else 0
            
            # Save best model state if model is provided
            if model is not None and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
                
            if self.verbose:
                print(f"✅ Validation loss improved to {val_loss:.6f}")
                
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} epochs")
                
            # Check if we should stop
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                    
        return self.early_stop
    
    def restore_best_model(self, model):
        """Restore the best model weights."""
        if self.best_model_state is not None and self.restore_best_weights:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"🔄 Restored best model weights from epoch {self.best_epoch}")
        else:
            if self.verbose:
                print("⚠️  No best model state to restore")
    
    def get_best_score(self):
        """Get the best validation loss achieved."""
        return self.best_loss
    
    def get_best_epoch(self):
        """Get the epoch where best validation loss was achieved."""
        return self.best_epoch
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None
        if self.verbose:
            print("🔄 Early stopping state reset")
