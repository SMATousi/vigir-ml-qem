#!/usr/bin/env python3
"""
Simple script to run the Optuna optimization for CLIP transformer.
This will save results to JSON and skip the final model training.
"""

import sys
import os

# Add current directory to path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_with_rand_data_optuna import train_with_rand_data_optuna

def main():
    """Run the Optuna optimization."""
    print("Starting CLIP Transformer Optuna Optimization")
    print("=" * 50)
    
    # Run optimization with custom parameters
    study = train_with_rand_data_optuna(
        n_trials=5,  # Adjust as needed
        timeout=3600,  # 1 hour timeout
        study_name="clip_transformer_optimization"
    )
    
    print("\nOptimization completed!")
    print("Results have been saved to JSON file.")

if __name__ == "__main__":
    main()
