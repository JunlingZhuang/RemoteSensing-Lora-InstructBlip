#!/usr/bin/env python3
"""
Evaluate models on RSIEval VQA dataset.

USAGE COMMANDS:

1. Evaluate native InstructBLIP (baseline):
   python evaluate_rsieval.py --model-type instructblip

2. Evaluate native BLIP-2 (baseline):
   python evaluate_rsieval.py --model-type blip2

3. Evaluate custom LoRA model:
   python evaluate_rsieval.py --model-type lora --lora-checkpoint path/to/checkpoint.pth

4. Quick test with limited samples:
   python evaluate_rsieval.py --model-type lora --lora-checkpoint path/to/checkpoint.pth --max-samples 50

5. Custom output path:
   python evaluate_rsieval.py --model-type lora --lora-checkpoint path/to/checkpoint.pth -o results/my_results.json

6. Custom RSIEval dataset path:
   python evaluate_rsieval.py --rsieval-path /path/to/RSIEval --model-type instructblip

EXAMPLES:

# Evaluate Ultra Conservative LoRA model
python evaluate_rsieval.py --model-type lora --lora-checkpoint checkpoints/ultra_conservative_epoch_10.pth

# Quick test of V5 Memory Optimized model
python evaluate_rsieval.py --model-type lora --lora-checkpoint checkpoints/grid_v5_memory_optimized_epoch_12.pth --max-samples 100

# Compare with native InstructBLIP baseline
python evaluate_rsieval.py --model-type instructblip --max-samples 100

OUTPUT:
- Accuracy by question type (scene, object, attribute, etc.)
- Overall accuracy percentage
- Sample VQA results
- Detailed JSON results file

"""

import os
import json
import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Add module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'module'))

from config import Config
from data.rsicap_dataset import load_rsieval_vqa_data
from inference.inferencer import ModelInferencer

# Initialize SentenceTransformer model for semantic similarity
SEMANTIC_MODEL = None

def get_semantic_model():
    """Lazy loading of SentenceTransformer model"""
    global SEMANTIC_MODEL
    if SEMANTIC_MODEL is None:
        print("Loading SentenceTransformer model for semantic similarity...")
        SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SEMANTIC_MODEL

def evaluate_answer_correctness(predicted_answer, ground_truth_answer, question_type):
    """
    Evaluate answer correctness using targeted approaches for different RSIEval categories.

    Args:
        predicted_answer: Generated answer from model
        ground_truth_answer: Ground truth answer
        question_type: Type of question (e.g., 'presence', 'quantity', etc.)

    Returns:
        bool: True if answer is correct, False otherwise
    """
    # Clean answers
    pred_clean = predicted_answer.lower().strip().rstrip('.')
    gt_clean = ground_truth_answer.lower().strip().rstrip('.')

    # For yes/no questions, use string processing and synonym matching
    if is_yes_no_question(gt_clean):
        return evaluate_yes_no_answer(pred_clean, gt_clean)

    # For simple numeric or short answers, use exact/substring matching
    if len(gt_clean.split()) <= 2 or gt_clean.isdigit():
        return evaluate_simple_answer(pred_clean, gt_clean)

    # For complex reasoning questions, use semantic similarity
    return evaluate_semantic_similarity(pred_clean, gt_clean)

def is_yes_no_question(answer):
    """Check if answer is yes/no type"""
    yes_no_words = {'yes', 'no', 'true', 'false'}
    return answer.lower().strip() in yes_no_words

def evaluate_yes_no_answer(predicted, ground_truth):
    """Evaluate yes/no answers with synonym matching"""
    # Define synonyms
    yes_synonyms = {'yes', 'true', 'correct', 'right', 'positive', 'affirmative'}
    no_synonyms = {'no', 'false', 'incorrect', 'wrong', 'negative'}

    # Check if ground truth is yes or no
    gt_is_yes = any(syn in ground_truth for syn in yes_synonyms)
    gt_is_no = any(syn in ground_truth for syn in no_synonyms)

    # Check predicted answer
    pred_is_yes = any(syn in predicted for syn in yes_synonyms)
    pred_is_no = any(syn in predicted for syn in no_synonyms)

    # Match logic
    if gt_is_yes and pred_is_yes:
        return True
    if gt_is_no and pred_is_no:
        return True

    return False

def evaluate_simple_answer(predicted, ground_truth):
    """Evaluate simple answers using substring matching"""
    # Try exact match first
    if predicted == ground_truth:
        return True

    # Try substring matching (ground truth in predicted)
    if ground_truth in predicted:
        return True

    # Try reverse substring matching (predicted in ground truth)
    if predicted in ground_truth:
        return True

    return False

def evaluate_semantic_similarity(predicted, ground_truth, threshold=0.75):
    """
    Evaluate semantic similarity using SentenceTransformers.

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        threshold: Similarity threshold (default 0.75)

    Returns:
        bool: True if similarity >= threshold
    """
    try:
        model = get_semantic_model()

        # Get embeddings (384-dimensional for all-MiniLM-L6-v2)
        embeddings = model.encode([predicted, ground_truth])
        pred_embedding = embeddings[0]
        gt_embedding = embeddings[1]

        # Calculate cosine similarity
        similarity = np.dot(pred_embedding, gt_embedding) / (
            np.linalg.norm(pred_embedding) * np.linalg.norm(gt_embedding)
        )

        return similarity >= threshold

    except Exception as e:
        print(f"Warning: Semantic similarity evaluation failed: {e}")
        # Fallback to simple string matching
        return evaluate_simple_answer(predicted, ground_truth)

def get_evaluation_method(ground_truth_answer, question_type):
    """Get the evaluation method used for debugging purposes"""
    gt_clean = ground_truth_answer.lower().strip().rstrip('.')

    if is_yes_no_question(gt_clean):
        return "yes_no_matching"
    elif len(gt_clean.split()) <= 2 or gt_clean.isdigit():
        return "simple_matching"
    else:
        return "semantic_similarity"


def fix_lora_adapter_config(checkpoint_path):
    """
    统一修复 LoRA adapter_config.json 文件中的不兼容参数

    Args:
        checkpoint_path: LoRA checkpoint 路径
    """
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"⚠️  adapter_config.json not found at {adapter_config_path}")
        return False

    try:
        # 读取当前配置
        with open(adapter_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 定义需要移除的不兼容参数
        invalid_params = [
            'corda_config', 'eva_config', 'exclude_modules', 'layer_replication',
            'layers_pattern', 'layers_to_transform', 'megatron_config', 'megatron_core',
            'qalora_group_size', 'trainable_token_indices', 'use_dora', 'use_qalora',
            'use_rslora', 'loftq_config', 'alpha_pattern', 'rank_pattern', 'lora_bias'
        ]

        # 检查并移除无效参数
        removed_params = []
        for param in invalid_params:
            if param in config:
                removed_params.append(param)
                del config[param]

        # If parameters were removed, save the fixed configuration
        if removed_params:
            print(f"Fixing adapter_config.json...")
            print(f"   Removing invalid parameters: {', '.join(removed_params)}")

            # Backup original file
            backup_path = adapter_config_path + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(adapter_config_path, backup_path)
                print(f"   Backup saved to: {backup_path}")

            # Save fixed configuration
            with open(adapter_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"adapter_config.json fixed successfully!")
            return True
        else:
            print(f"adapter_config.json is already compatible")
            return False

    except Exception as e:
        print(f"Error fixing adapter_config.json: {e}")
        return False


def fix_all_lora_configs_in_directory(checkpoint_dir):
    """
    Fix all possible LoRA configuration files in directory

    Args:
        checkpoint_dir: checkpoint directory path
    """
    print(f"Scanning for LoRA configs in: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    fixed_count = 0

    # Traverse all subdirectories and files in the directory
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file == "adapter_config.json":
                config_path = os.path.join(root, file)
                print(f"   Found adapter_config.json: {config_path}")

                # Fix this configuration file
                parent_dir = os.path.dirname(config_path)
                if fix_lora_adapter_config(parent_dir):
                    fixed_count += 1

    if fixed_count > 0:
        print(f"Fixed {fixed_count} LoRA configuration file(s)")
    else:
        print(f"All LoRA configurations are compatible")


def evaluate_model_vqa_on_rsieval(rsieval_path, model_type="instructblip", lora_checkpoint=None, output_path=None, max_samples=None):
    """Evaluate models on RSIEval VQA dataset
    
    Args:
        rsieval_path: Path to RSIEval dataset
        model_type: One of ["instructblip", "blip2", "lora"]
        lora_checkpoint: Path to LoRA checkpoint (required if model_type="lora")
        output_path: Output JSON file path
        max_samples: Limit number of samples for testing
    """
    
    # Create config
    config = Config()
    
    print("Loading RSIEval VQA dataset...")
    test_loader, processor, qa_samples = load_rsieval_vqa_data(config, rsieval_path)
    
    if max_samples:
        # Limit samples for testing
        limited_samples = qa_samples[:max_samples]
        print(f"Limiting to {max_samples} QA pairs for testing")
    else:
        limited_samples = qa_samples
    
    # Initialize model based on type
    if model_type == "lora":
        if not lora_checkpoint:
            raise ValueError("LoRA checkpoint path required for model_type='lora'")

        # Uniformly fix LoRA configuration files
        print(f"Checking LoRA checkpoint: {lora_checkpoint}")

        # If it's a single checkpoint, fix directly
        if os.path.isdir(lora_checkpoint):
            fix_lora_adapter_config(lora_checkpoint)
        else:
            # If it's the parent directory of checkpoint directory, scan all subdirectories
            checkpoint_parent = os.path.dirname(lora_checkpoint)
            if os.path.exists(checkpoint_parent):
                fix_all_lora_configs_in_directory(checkpoint_parent)

        print(f"Initializing LoRA model from checkpoint: {lora_checkpoint}")
        inferencer = ModelInferencer(model_type="lora", model_path=lora_checkpoint)
        model_name = f"lora_{os.path.basename(lora_checkpoint)}"
    elif model_type == "blip2":
        print("Initializing native BLIP-2 model...")
        inferencer = ModelInferencer(model_type="blip2")
        model_name = "native_blip2"
    else:  # default to instructblip
        print("Initializing native InstructBLIP model...")
        inferencer = ModelInferencer(model_type="instructblip")
        model_name = "native_instructblip"
    
    results = []
    correct_answers = 0

    # Track accuracy by question type
    type_stats = {}

    # Track evaluation method usage
    evaluation_method_stats = {
        "yes_no_matching": {"total": 0, "correct": 0},
        "simple_matching": {"total": 0, "correct": 0},
        "semantic_similarity": {"total": 0, "correct": 0}
    }
    
    print(f"Running VQA inference on {len(limited_samples)} QA pairs...")
    for i, sample in enumerate(tqdm(limited_samples)):
        image_path = os.path.join(test_loader.dataset.images_dir, sample['filename'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Generate answer using the question as prompt
            generated_answer = inferencer.generate_caption(
                image_path,
                sample['question'],  # Use question as prompt
                max_new_tokens=300,  # RSGPT用这个
                num_beams=1,         # RSGPT对话用1
                do_sample=True,      # RSGPT对话用True
                top_p=0.9,
                repetition_penalty=1.0,  # RSGPT对话用1.0
                temperature=1.0
            )

            # Clear GPU cache periodically to prevent memory accumulation
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                print(f"GPU cache cleared at sample {i + 1}")
            
            # Evaluate answer correctness using targeted approaches
            is_correct = evaluate_answer_correctness(
                generated_answer,
                sample['answer'],
                sample['type']
            )
            if is_correct:
                correct_answers += 1

            # Update type statistics
            question_type = sample['type']
            if question_type not in type_stats:
                type_stats[question_type] = {'correct': 0, 'total': 0}
            type_stats[question_type]['total'] += 1
            if is_correct:
                type_stats[question_type]['correct'] += 1

            # Update evaluation method statistics
            eval_method = get_evaluation_method(sample['answer'], sample['type'])
            evaluation_method_stats[eval_method]['total'] += 1
            if is_correct:
                evaluation_method_stats[eval_method]['correct'] += 1
            
            result = {
                "sample_id": i,
                "image_file": sample['filename'],
                "question": sample['question'],
                "ground_truth_answer": sample['answer'],
                "generated_answer": generated_answer,
                "question_type": sample['type'],
                "is_correct": is_correct,
                "evaluation_method": get_evaluation_method(sample['answer'], sample['type'])
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                accuracy = correct_answers / len(results) * 100
                print(f"Processed {i + 1}/{len(limited_samples)} QA pairs")
                print(f"Current accuracy: {accuracy:.2f}%")
                print(f"Latest Q: {sample['question']}")
                print(f"Ground truth: {sample['answer']}")
                print(f"Generated: {generated_answer}")
                print("-" * 40)
        
        except Exception as e:
            print(f"Error processing sample {i} ({sample['filename']}): {e}")
            print(f"   Question: {sample['question']}")

            # Save partial results if we've processed a significant number
            if len(results) >= 50 and (i + 1) % 100 == 0:
                partial_output_path = output_path.replace('.json', f'_partial_{i+1}.json') if output_path else f"partial_results_{i+1}.json"
                print(f"Saving partial results to {partial_output_path}")

                partial_data = {
                    "model_type": model_name,
                    "dataset": "RSIEval_VQA",
                    "processed_samples": i + 1,
                    "successful_samples": len(results),
                    "partial_accuracy": correct_answers / len(results) * 100 if results else 0,
                    "results": results
                }

                with open(partial_output_path, 'w', encoding='utf-8') as f:
                    json.dump(partial_data, f, ensure_ascii=False, indent=2)

            continue
    
    # Calculate final accuracy and type accuracies
    final_accuracy = correct_answers / len(results) * 100 if results else 0
    
    print(f"\nCompleted VQA inference on {len(results)}/{len(limited_samples)} QA pairs")
    print(f"\n{'='*60}")
    print("RESULTS BY QUESTION TYPE")
    print(f"{'='*60}")
    
    type_accuracies = {}
    for qtype, stats in type_stats.items():
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        type_accuracies[qtype] = accuracy
        print(f"{qtype:15s}: {accuracy:6.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    
    print(f"{'='*60}")
    print(f"Overall Accuracy: {final_accuracy:.2f}% ({correct_answers}/{len(results)})")
    print(f"{'='*60}")

    # Print evaluation method statistics
    print("\nEVALUATION METHOD BREAKDOWN")
    print(f"{'='*60}")
    for method, stats in evaluation_method_stats.items():
        if stats['total'] > 0:
            method_accuracy = stats['correct'] / stats['total'] * 100
            print(f"{method:20s}: {method_accuracy:6.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    print(f"{'='*60}")
    
    # Save results
    if output_path:
        # Create results directory
        results_dir = os.path.dirname(output_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        
        output_data = {
            "model_type": model_name,
            "model_config": {
                "type": model_type,
                "checkpoint": lora_checkpoint if model_type == "lora" else None
            },
            "dataset": "RSIEval_VQA",
            "total_qa_pairs": len(limited_samples),
            "successful_qa_pairs": len(results),
            "overall_accuracy": final_accuracy,
            "correct_answers": correct_answers,
            "accuracy_by_type": type_accuracies,
            "type_statistics": type_stats,
            "evaluation_method_statistics": evaluation_method_stats,
            "generation_config": {
                "max_new_tokens": 300,
                "num_beams": 1,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "temperature": 1.0
            },
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    # Show sample results
    print("\n" + "="*60)
    print("SAMPLE VQA RESULTS")
    print("="*60)
    for i, result in enumerate(results[:3]):
        print(f"\nSample {i+1}:")
        print(f"Image: {result['image_file']}")
        print(f"Question: {result['question']}")
        print(f"Ground Truth Answer: {result['ground_truth_answer']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"Correct: {result['is_correct']}")
        print("-" * 40)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on RSIEval VQA dataset")
    
    parser.add_argument(
        "--rsieval-path",
        type=str,
        required=False,
        default=None,
        help="Path to RSIEval dataset directory (default: data/rsgpt_dataset/RSIEval)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["instructblip", "blip2", "lora"],
        default="instructblip",
        help="Model type to evaluate"
    )
    
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        help="Path to LoRA checkpoint (required for --model-type lora)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of QA pairs to evaluate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == "lora" and not args.lora_checkpoint:
        print("Error: --lora-checkpoint is required when using --model-type lora")
        sys.exit(1)
    
    # Use default path if not provided
    if args.rsieval_path is None:
        args.rsieval_path = "data/rsgpt_dataset/RSIEval"

    if not os.path.exists(args.rsieval_path):
        print(f"Error: RSIEval path not found: {args.rsieval_path}")
        sys.exit(1)
    
    # Auto-generate output filename if not provided
    if not args.output:
        # Create results directory
        results_dir = "results"
        if args.model_type == "lora":
            checkpoint_name = os.path.splitext(os.path.basename(args.lora_checkpoint))[0]
            args.output = os.path.join(results_dir, f"rsieval_{args.model_type}_{checkpoint_name}_vqa_results.json")
        else:
            args.output = os.path.join(results_dir, f"rsieval_{args.model_type}_vqa_results.json")
    
    try:
        results = evaluate_model_vqa_on_rsieval(
            rsieval_path=args.rsieval_path,
            model_type=args.model_type,
            lora_checkpoint=args.lora_checkpoint,
            output_path=args.output,
            max_samples=args.max_samples
        )
        
        print(f"\nVQA Evaluation completed successfully!")
        print(f"Model: {args.model_type}")
        if args.lora_checkpoint:
            print(f"LoRA checkpoint: {args.lora_checkpoint}")
        print(f"Total QA pairs processed: {len(results)}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"VQA Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()