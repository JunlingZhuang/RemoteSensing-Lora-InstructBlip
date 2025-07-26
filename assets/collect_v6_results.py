#!/usr/bin/env python3
"""
Collect all v6 experiment results for analysis
"""
import json
import os
from pathlib import Path

def collect_v6_results():
    """Collect all v6 experiment results"""
    checkpoints_dir = Path("checkpoints")
    results = []
    
    # Find all v6 augmented experiments
    v6_experiments = [
        "grid_v6_exp1_r16_a64_d05_cosine_augmented_20250725_093609",
        "grid_v6_exp2_r16_a48_d10_linear_augmented_20250725_113819", 
        "grid_v6_exp3_r24_a64_d10_linear_augmented_20250725_134034",
        "grid_v6_exp4_r24_a48_d05_cosine_augmented_20250725_154933",
        "grid_v6_exp5_r32_a64_d05_linear_augmented_20250725_175830",
        "grid_v6_exp6_r32_a32_d10_cosine_augmented_20250725_200755",
        "grid_v6_exp7_r16_a48_d10_cosine_augmented_20250725_221710",
        "grid_v6_exp8_r32_a48_d05_linear_augmented_20250726_002621"
    ]
    
    for exp_name in v6_experiments:
        exp_dir = checkpoints_dir / exp_name
        summary_file = exp_dir / "training_summary_detailed.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
            # Extract key information
            config = data.get('config', {})
            result = {
                'experiment': exp_name.split('_')[2],  # exp1, exp2, etc.
                'lora_r': config.get('lora_r'),
                'lora_alpha': config.get('lora_alpha'), 
                'lora_dropout': config.get('lora_dropout'),
                'scheduler': config.get('scheduler_type'),
                'best_val_loss': data.get('best_val_loss'),
                'final_val_loss': data.get('final_val_loss'),
                'best_epoch': data.get('best_val_epoch'),
                'total_epochs': data.get('total_epochs'),
                'training_time': data.get('total_training_time'),
                'avg_epoch_time': data.get('avg_epoch_time')
            }
            results.append(result)
            print(f"[OK] {result['experiment']}: r={result['lora_r']}, α={result['lora_alpha']}, {result['scheduler']}, loss={result['best_val_loss']:.4f}")
        else:
            print(f"[MISSING] {exp_name}")
    
    # Sort by best validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    # Save results
    with open('assets/v6_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = collect_v6_results()
    print(f"\nCollected {len(results)} experiment results")
    print("Top 3 performers:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result['experiment']}: {result['best_val_loss']:.4f}")
