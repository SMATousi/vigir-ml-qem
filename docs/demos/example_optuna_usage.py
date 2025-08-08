#!/usr/bin/env python3
"""
Example script demonstrating how to use the train_with_rand_data_optuna function
to optimize CLIP transformer architecture for quantum error mitigation.
"""

from train_with_rand_data_optuna import train_with_rand_data_optuna, train_best_model_from_study

def main():
    """
    Main function to run Optuna optimization for CLIP transformer.
    """
    print("Starting CLIP Transformer Architecture Optimization with Optuna")
    print("=" * 60)
    
    # Run Optuna optimization
    # Adjust n_trials and timeout based on your computational resources
    study = train_with_rand_data_optuna(
        n_trials=50,  # Start with fewer trials for testing
        timeout=1800,  # 30 minutes timeout
        study_name="clip_transformer_qem_optimization"
    )
    
    # Train final models with best parameters
    if study.best_trial is not None:
        print("\nTraining final models with optimized hyperparameters...")
        models, metrics = train_best_model_from_study(
            study, 
            save_path="optimized_clip_models"
        )
        
        print("\nOptimization completed successfully!")
        print(f"Best models saved in: optimized_clip_models/")
        print(f"Final metrics: {metrics}")
        
    else:
        print("Optimization failed - no successful trials completed.")
        print("Consider:")
        print("1. Increasing the timeout")
        print("2. Checking data file availability")
        print("3. Reducing the complexity of hyperparameter space")

if __name__ == "__main__":
    main()
