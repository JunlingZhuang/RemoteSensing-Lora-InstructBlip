#!/usr/bin/env python3
"""
Evaluate native InstructBLIP on RSIEval VQA dataset.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'module'))

from config import Config
from data.rsicap_dataset import load_rsieval_vqa_data
from inference.inferencer import ModelInferencer


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
            
            # Simple answer matching (case-insensitive)
            # Remove punctuation for better matching
            gt_clean = sample['answer'].lower().strip().rstrip('.')
            gen_clean = generated_answer.lower().strip()
            is_correct = gt_clean in gen_clean
            if is_correct:
                correct_answers += 1
            
            # Update type statistics
            question_type = sample['type']
            if question_type not in type_stats:
                type_stats[question_type] = {'correct': 0, 'total': 0}
            type_stats[question_type]['total'] += 1
            if is_correct:
                type_stats[question_type]['correct'] += 1
            
            result = {
                "sample_id": i,
                "image_file": sample['filename'],
                "question": sample['question'],
                "ground_truth_answer": sample['answer'],
                "generated_answer": generated_answer,
                "question_type": sample['type'],
                "is_correct": is_correct
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
            print(f"Error processing sample {i}: {e}")
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