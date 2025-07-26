#!/usr/bin/env python3
"""
RSIEval Dataset Analysis and Visualization
Analyze RSIEval VQA dataset distribution, including:
1. Question type distribution
2. Answer length distribution
3. Question length distribution
4. Number of QA pairs per image
5. Answer type analysis (yes/no, numeric, descriptive, etc.)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import re
from tqdm import tqdm

# Try to import optional dependencies
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set matplotlib backend and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if HAS_SEABORN:
    plt.style.use('seaborn-v0_8')
else:
    plt.style.use('default')

def load_rsieval_data():
    """Load RSIEval dataset"""
    rsieval_file = 'data/rsgpt_dataset/RSIEval/annotations.json'
    images_dir = 'data/rsgpt_dataset/RSIEval/images'

    with open(rsieval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['annotations'], images_dir

def analyze_qa_distribution(annotations):
    """Analyze QA pair distribution"""

    # Statistical information
    total_images = len(annotations)
    total_qa_pairs = sum(len(ann['qa_pairs']) for ann in annotations)

    # Statistics by type
    type_counts = Counter()
    qa_per_image = []
    question_lengths = []
    answer_lengths = []
    all_questions = []
    all_answers = []

    # Answer type analysis
    yes_no_answers = Counter()
    numeric_answers = []
    descriptive_answers = []
    
    for annotation in annotations:
        qa_pairs = annotation['qa_pairs']
        qa_per_image.append(len(qa_pairs))
        
        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            qa_type = qa['type']
            
            # Basic statistics
            type_counts[qa_type] += 1
            question_lengths.append(len(question))
            answer_lengths.append(len(answer))
            all_questions.append(question)
            all_answers.append(answer)

            # Answer type analysis
            answer_lower = answer.lower().strip()
            if answer_lower in ['yes', 'no']:
                yes_no_answers[answer_lower] += 1
            elif re.match(r'^\d+$', answer_lower):
                numeric_answers.append(int(answer_lower))
            else:
                descriptive_answers.append(answer)
    
    # 构建分析结果
    analysis_results = {
        'dataset_overview': {
            'total_images': total_images,
            'total_qa_pairs': total_qa_pairs,
            'avg_qa_per_image': np.mean(qa_per_image),
            'qa_per_image_distribution': {
                'min': min(qa_per_image),
                'max': max(qa_per_image),
                'mean': np.mean(qa_per_image),
                'std': np.std(qa_per_image)
            }
        },
        'question_type_distribution': dict(type_counts),
        'question_analysis': {
            'avg_length': np.mean(question_lengths),
            'length_distribution': {
                'min': min(question_lengths),
                'max': max(question_lengths),
                'mean': np.mean(question_lengths),
                'std': np.std(question_lengths)
            }
        },
        'answer_analysis': {
            'avg_length': np.mean(answer_lengths),
            'length_distribution': {
                'min': min(answer_lengths),
                'max': max(answer_lengths),
                'mean': np.mean(answer_lengths),
                'std': np.std(answer_lengths)
            },
            'yes_no_distribution': dict(yes_no_answers),
            'numeric_answers_count': len(numeric_answers),
            'descriptive_answers_count': len(descriptive_answers)
        }
    }
    
    return analysis_results, {
        'qa_per_image': qa_per_image,
        'question_lengths': question_lengths,
        'answer_lengths': answer_lengths,
        'all_questions': all_questions,
        'all_answers': all_answers,
        'numeric_answers': numeric_answers,
        'descriptive_answers': descriptive_answers
    }

def analyze_question_patterns(questions):
    """Analyze question patterns"""

    # Question starter word statistics
    question_starters = Counter()
    question_keywords = Counter()

    for question in questions:
        words = question.lower().split()
        if words:
            question_starters[words[0]] += 1

        # Extract keywords
        keywords = ['what', 'where', 'how', 'why', 'when', 'which', 'who', 'is', 'are', 'can', 'do', 'does']
        for keyword in keywords:
            if keyword in question.lower():
                question_keywords[keyword] += 1
    
    return {
        'question_starters': dict(question_starters.most_common(20)),
        'question_keywords': dict(question_keywords)
    }

def create_visualizations(analysis_results, raw_data):
    """Create visualization charts"""

    # Create output directory
    assets_dir = 'assets'
    os.makedirs(assets_dir, exist_ok=True)

    # 1. 主要分布统计图 (与RSICap样式一致)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RSIEval Dataset Statistics', fontsize=16, fontweight='bold')

    # 子图1: 问题类型分布
    types = list(analysis_results['question_type_distribution'].keys())
    counts = list(analysis_results['question_type_distribution'].values())

    axes[0,0].bar(types, counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Question Type Distribution')
    axes[0,0].set_xlabel('Question Type')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for i, (bar_type, count) in enumerate(zip(types, counts)):
        axes[0,0].text(i, count + max(counts)*0.01, str(count),
                      ha='center', va='bottom', fontweight='bold')
    
    # 子图2: QA对每张图片分布
    axes[0,1].hist(raw_data['qa_per_image'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('QA Pairs per Image Distribution')
    axes[0,1].set_xlabel('Number of QA Pairs')
    axes[0,1].set_ylabel('Number of Images')
    axes[0,1].axvline(analysis_results['dataset_overview']['avg_qa_per_image'],
                     color='red', linestyle='--',
                     label=f'Mean: {analysis_results["dataset_overview"]["avg_qa_per_image"]:.1f}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 子图3: 问题长度分布
    axes[1,0].hist(raw_data['question_lengths'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Question Length Distribution')
    axes[1,0].set_xlabel('Characters')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(analysis_results['question_analysis']['avg_length'],
                     color='red', linestyle='--',
                     label=f'Mean: {analysis_results["question_analysis"]["avg_length"]:.1f}')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 子图4: 统计摘要 (与RSICap样式一致)
    axes[1,1].axis('off')
    summary_text = f"""
    Dataset Summary:
    • Total Images: {analysis_results['dataset_overview']['total_images']:,}
    • Total QA Pairs: {analysis_results['dataset_overview']['total_qa_pairs']:,}
    • Avg QA per Image: {analysis_results['dataset_overview']['avg_qa_per_image']:.1f}
    • Question Types: {len(analysis_results['question_type_distribution'])}

    Question Length:
    • Min: {analysis_results['question_analysis']['length_distribution']['min']} chars
    • Max: {analysis_results['question_analysis']['length_distribution']['max']} chars
    • Avg: {analysis_results['question_analysis']['avg_length']:.1f} chars

    Answer Length:
    • Min: {analysis_results['answer_analysis']['length_distribution']['min']} chars
    • Max: {analysis_results['answer_analysis']['length_distribution']['max']} chars
    • Avg: {analysis_results['answer_analysis']['avg_length']:.1f} chars
    """
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{assets_dir}/rsieval_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 详细分析图 (与RSICap词频分析样式一致)
    plt.figure(figsize=(15, 10))

    # 问题类型饼图
    plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    plt.pie(counts, labels=types, autopct='%1.1f%%', colors=colors[:len(types)], startangle=90)
    plt.title('Question Type Distribution', fontsize=14, fontweight='bold')

    # 答案类型分析
    plt.subplot(2, 2, 2)
    answer_types = ['Yes/No', 'Numeric', 'Descriptive']
    answer_type_counts = [
        sum(analysis_results['answer_analysis']['yes_no_distribution'].values()),
        analysis_results['answer_analysis']['numeric_answers_count'],
        analysis_results['answer_analysis']['descriptive_answers_count']
    ]

    bars = plt.bar(answer_types, answer_type_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Answer Type Distribution')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, count in zip(bars, answer_type_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(answer_type_counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')

    # Yes/No答案分布
    plt.subplot(2, 2, 3)
    if analysis_results['answer_analysis']['yes_no_distribution']:
        yes_no_labels = list(analysis_results['answer_analysis']['yes_no_distribution'].keys())
        yes_no_counts = list(analysis_results['answer_analysis']['yes_no_distribution'].values())
        bars = plt.bar(yes_no_labels, yes_no_counts, color=['#2ECC71', '#E74C3C'])
        plt.title('Yes/No Answer Distribution')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        for bar, count in zip(bars, yes_no_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(yes_no_counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')

    # 答案长度分布
    plt.subplot(2, 2, 4)
    plt.hist(raw_data['answer_lengths'], bins=25, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Answer Length Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.axvline(analysis_results['answer_analysis']['avg_length'],
                color='red', linestyle='--',
                label=f'Mean: {analysis_results["answer_analysis"]["avg_length"]:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{assets_dir}/rsieval_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return assets_dir

def save_analysis_results(analysis_results, question_patterns, output_dir='assets'):
    """Save analysis results to JSON file"""

    # Merge all analysis results
    complete_results = {
        'dataset_name': 'RSIEval',
        'analysis_timestamp': '2024-07-25',
        'dataset_statistics': analysis_results,
        'question_patterns': question_patterns,
        'summary': {
            'total_question_types': len(analysis_results['question_type_distribution']),
            'most_common_type': max(analysis_results['question_type_distribution'].items(), key=lambda x: x[1]),
            'least_common_type': min(analysis_results['question_type_distribution'].items(), key=lambda x: x[1])
        }
    }
    
    # 保存为JSON
    output_file = os.path.join(output_dir, 'rsieval_analysis_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis results saved to {output_file}")
    return output_file

def main():
    """Main function"""
    print("Starting RSIEval Dataset Analysis...")

    # Load data
    annotations, images_dir = load_rsieval_data()
    print(f"Loaded {len(annotations)} images with QA pairs")

    # Analyze QA distribution
    print("Analyzing QA distribution...")
    analysis_results, raw_data = analyze_qa_distribution(annotations)

    # Analyze question patterns
    print("Analyzing question patterns...")
    question_patterns = analyze_question_patterns(raw_data['all_questions'])

    # Create visualizations
    print("Creating visualizations...")
    assets_dir = create_visualizations(analysis_results, raw_data)

    # Save analysis results
    print("Saving analysis results...")
    save_analysis_results(analysis_results, question_patterns, assets_dir)

    # Print summary
    print("\n" + "="*50)
    print("RSIEval Dataset Summary")
    print("="*50)
    print(f"Total Images: {analysis_results['dataset_overview']['total_images']:,}")
    print(f"Total QA Pairs: {analysis_results['dataset_overview']['total_qa_pairs']:,}")
    print(f"Avg QA per Image: {analysis_results['dataset_overview']['avg_qa_per_image']:.1f}")
    print(f"Question Types: {len(analysis_results['question_type_distribution'])}")
    
    print("\nQuestion Type Distribution:")
    for qtype, count in sorted(analysis_results['question_type_distribution'].items(), 
                              key=lambda x: x[1], reverse=True):
        percentage = (count / analysis_results['dataset_overview']['total_qa_pairs']) * 100
        print(f"  {qtype}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nAnalysis complete! Results saved to {assets_dir}/")

if __name__ == "__main__":
    main()
