#!/usr/bin/env python3
"""
Test script to verify wandb integration with SimpleTransformerEstimator
"""

import numpy as np
import torch
import pandas as pd
from simple_transformer import SimpleTransformerEstimator

# Check if wandb is available
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ wandb is available")
except ImportError:
    print("✗ wandb not installed. Please install with: pip install wandb")
    WANDB_AVAILABLE = False

def test_wandb_integration():
    """Test the wandb integration with a small dataset"""
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    seq_len = 170
    
    X_train = np.random.randn(n_samples, seq_len).astype(np.float32)
    y_train = np.random.randn(n_samples)
    
    X_test = np.random.randn(50, seq_len).astype(np.float32)
    y_test = np.random.randn(50)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    if WANDB_AVAILABLE:
        # Initialize wandb
        wandb.init(project="vigir-ml-qem-test", 
                   name="test_run",
                   config={
                       "n_epochs": 10,
                       "model_type": "test",
                       "seq_len": seq_len
                   })
    
    # Create model with wandb logging
    model = SimpleTransformerEstimator(
        n_epochs=10,
        d_model=32,
        nhead=4,
        lr=0.001,
        verbose=True,
        seq_len=seq_len,
        wandb_logging=WANDB_AVAILABLE,
        eval_data=(X_test, y_test),
        qubit_idx=0
    )
    
    print("Starting training with wandb logging...")
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    
    if WANDB_AVAILABLE:
        wandb.log({"final_test_rmse": rmse})
        wandb.finish()
        print("✓ wandb logging completed successfully")
    
    print("Test completed!")

if __name__ == "__main__":
    test_wandb_integration()
