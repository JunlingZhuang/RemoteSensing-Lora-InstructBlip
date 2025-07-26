#!/usr/bin/env python3
"""
Organize training curve images for the 8 LoRA experiments
"""
import os
import shutil
from pathlib import Path

def organize_training_curves():
    """Copy and rename training curve images"""
    
    # Create output directory
    output_dir = Path("latex/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from experiment to figure name and model description
    experiment_mapping = {
        # Based on performance ranking from v6 results
        "grid_v6_exp3_r24_a64_d10_linear_augmented_20250725_134034": {
            "figure_name": "training_curve_a.png",
            "model_name": "RSI-LoRA-24r64a-Linear-Aug",
            "rank": 1,
            "description": "Best performer (1.2727)"
        },
        "grid_v6_exp5_r32_a64_d05_linear_augmented_20250725_175830": {
            "figure_name": "training_curve_b.png", 
            "model_name": "RSI-LoRA-32r64a-Linear-Aug",
            "rank": 2,
            "description": "Second best (1.2738)"
        },
        "grid_v6_exp8_r32_a48_d05_linear_augmented_20250726_002621": {
            "figure_name": "training_curve_c.png",
            "model_name": "RSI-LoRA-32r48a-Linear-Aug", 
            "rank": 3,
            "description": "Third best (1.2876)"
        },
        "grid_v6_exp7_r16_a48_d10_cosine_augmented_20250725_221710": {
            "figure_name": "training_curve_d.png",
            "model_name": "RSI-LoRA-16r48a-Cos-Aug",
            "rank": 4,
            "description": "Best cosine (1.2940)"
        },
        "grid_v6_exp2_r16_a48_d10_linear_augmented_20250725_113819": {
            "figure_name": "training_curve_e.png",
            "model_name": "RSI-LoRA-16r48a-Linear-Aug",
            "rank": 5,
            "description": "Mid-range linear (1.3053)"
        },
        "grid_v6_exp6_r32_a32_d10_cosine_augmented_20250725_200755": {
            "figure_name": "training_curve_f.png",
            "model_name": "RSI-LoRA-32r32a-Cos-Aug",
            "rank": 6,
            "description": "Low alpha cosine (1.3111)"
        },
        "grid_v6_exp1_r16_a64_d05_cosine_augmented_20250725_093609": {
            "figure_name": "training_curve_g.png",
            "model_name": "RSI-LoRA-16r64a-Cos-Aug",
            "rank": 7,
            "description": "High alpha cosine (1.3381)"
        },
        "grid_v6_exp4_r24_a48_d05_cosine_augmented_20250725_154933": {
            "figure_name": "training_curve_h.png",
            "model_name": "RSI-LoRA-24r48a-Cos-Aug",
            "rank": 8,
            "description": "Worst performer (1.3385)"
        }
    }
    
    checkpoints_dir = Path("checkpoints")
    copied_files = []
    missing_files = []
    
    print("Organizing training curve images...")
    print("=" * 60)
    
    for exp_dir, info in experiment_mapping.items():
        source_path = checkpoints_dir / exp_dir / "training_curves.png"
        target_path = output_dir / info["figure_name"]
        
        if source_path.exists():
            # Copy the file
            shutil.copy2(source_path, target_path)
            copied_files.append({
                "source": str(source_path),
                "target": str(target_path),
                "model": info["model_name"],
                "rank": info["rank"],
                "description": info["description"]
            })
            print(f"[OK] {info['figure_name']}: {info['model_name']} - {info['description']}")
        else:
            missing_files.append({
                "expected_source": str(source_path),
                "target": info["figure_name"],
                "model": info["model_name"]
            })
            print(f"[MISSING] {source_path}")
    
    print("\n" + "=" * 60)
    print(f"Successfully copied: {len(copied_files)} files")
    print(f"Missing files: {len(missing_files)} files")
    
    if copied_files:
        print(f"\nFiles saved to: {output_dir}/")
        print("\nFigure mapping for LaTeX:")
        print("-" * 40)
        for file_info in sorted(copied_files, key=lambda x: x["rank"]):
            print(f"{file_info['target'].split('/')[-1]}: {file_info['model']} (Rank {file_info['rank']})")
    
    if missing_files:
        print(f"\nMissing files:")
        for missing in missing_files:
            print(f"  - {missing['expected_source']}")
    
    # Create a summary file
    summary_path = output_dir / "figure_mapping.txt"
    with open(summary_path, 'w') as f:
        f.write("Training Curve Figure Mapping\n")
        f.write("=" * 40 + "\n\n")
        for file_info in sorted(copied_files, key=lambda x: x["rank"]):
            f.write(f"{file_info['target'].split('/')[-1]}: {file_info['model']} - {file_info['description']}\n")
        
        f.write(f"\nGenerated on: {Path.cwd()}\n")
        f.write(f"Total figures: {len(copied_files)}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    return copied_files, missing_files

if __name__ == "__main__":
    copied, missing = organize_training_curves()
