#!/usr/bin/env python3
"""
RSICap Dataset Analysis and Visualization
Generate detailed analysis charts for the dataset, including:
1. Text length distribution
2. Word frequency analysis
3. Image size analysis
4. Keyword analysis
5. Sentence structure analysis
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from PIL import Image
import re
from tqdm import tqdm

# Try to import optional dependencies
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Set font and style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if HAS_SEABORN:
    plt.style.use('seaborn-v0_8')
else:
    plt.style.use('default')

def load_rsicap_data():
    """Load RSICap dataset"""
    rsicap_file = 'data/rsgpt_dataset/RSICap/captions.json'
    images_dir = 'data/rsgpt_dataset/RSICap/images'

    with open(rsicap_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['annotations'], images_dir

def analyze_text_statistics(annotations):
    """Analyze text statistics"""
    captions = [ann['caption'] for ann in annotations]

    # Basic statistics
    text_lengths = [len(caption) for caption in captions]
    word_counts = [len(caption.split()) for caption in captions]
    sentence_counts = [len(re.split(r'[.!?]+', caption)) - 1 for caption in captions]
    
    stats = {
        'total_samples': len(annotations),
        'avg_text_length': np.mean(text_lengths),
        'avg_word_count': np.mean(word_counts),
        'avg_sentence_count': np.mean(sentence_counts),
        'text_lengths': text_lengths,
        'word_counts': word_counts,
        'sentence_counts': sentence_counts
    }
    
    return stats, captions

def analyze_vocabulary(captions):
    """Analyze vocabulary statistics"""
    # Combine all text
    all_text = ' '.join(captions).lower()

    # Clean text
    words = re.findall(r'\b[a-zA-Z]+\b', all_text)

    # Word frequency statistics
    word_freq = Counter(words)

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'}
    
    filtered_word_freq = Counter({word: count for word, count in word_freq.items()
                                  if word not in stop_words and len(word) > 2})
    
    return word_freq, filtered_word_freq

def analyze_remote_sensing_keywords(captions):
    """Analyze remote sensing related keywords"""
    rs_keywords = {
        'objects': ['vehicle', 'car', 'truck', 'bus', 'building', 'house', 'tree', 'road', 'bridge', 'ship', 'boat', 'plane', 'aircraft'],
        'terrain': ['forest', 'field', 'water', 'river', 'lake', 'mountain', 'hill', 'valley', 'desert', 'beach', 'coast'],
        'urban': ['city', 'town', 'street', 'parking', 'lot', 'residential', 'commercial', 'industrial', 'urban', 'suburban'],
        'colors': ['green', 'blue', 'red', 'white', 'black', 'yellow', 'brown', 'gray', 'grey'],
        'spatial': ['left', 'right', 'top', 'bottom', 'center', 'middle', 'corner', 'edge', 'north', 'south', 'east', 'west']
    }
    
    keyword_counts = defaultdict(Counter)
    
    for caption in captions:
        caption_lower = caption.lower()
        for category, keywords in rs_keywords.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    keyword_counts[category][keyword] += caption_lower.count(keyword)
    
    return keyword_counts

def analyze_image_properties(annotations, images_dir):
    """Analyze image properties"""
    image_sizes = []
    file_sizes = []

    print("Analyzing image properties...")
    for ann in tqdm(annotations[:100]):  # Analyze first 100 images as sample
        image_path = os.path.join(images_dir, ann['filename'])
        if os.path.exists(image_path):
            try:
                # 图像尺寸
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
                
                # 文件大小
                file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
                file_sizes.append(file_size)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return image_sizes, file_sizes

def create_visualizations(stats, captions, word_freq, filtered_word_freq, keyword_counts, image_sizes, file_sizes):
    """Create visualization charts"""

    # Create output directory
    output_dir = 'rsicap_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 文本长度分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RSICap Dataset Text Statistics', fontsize=16, fontweight='bold')
    
    # 字符长度分布
    axes[0,0].hist(stats['text_lengths'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Character Length Distribution')
    axes[0,0].set_xlabel('Characters')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(stats['avg_text_length'], color='red', linestyle='--', 
                     label=f'Mean: {stats["avg_text_length"]:.1f}')
    axes[0,0].legend()
    
    # 词数分布
    axes[0,1].hist(stats['word_counts'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Word Count Distribution')
    axes[0,1].set_xlabel('Words')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(stats['avg_word_count'], color='red', linestyle='--',
                     label=f'Mean: {stats["avg_word_count"]:.1f}')
    axes[0,1].legend()
    
    # 句子数分布
    axes[1,0].hist(stats['sentence_counts'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Sentence Count Distribution')
    axes[1,0].set_xlabel('Sentences')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(stats['avg_sentence_count'], color='red', linestyle='--',
                     label=f'Mean: {stats["avg_sentence_count"]:.1f}')
    axes[1,0].legend()
    
    # 统计摘要
    axes[1,1].axis('off')
    summary_text = f"""
    Dataset Summary:
    • Total Samples: {stats['total_samples']:,}
    • Avg Characters: {stats['avg_text_length']:.1f}
    • Avg Words: {stats['avg_word_count']:.1f}
    • Avg Sentences: {stats['avg_sentence_count']:.1f}
    
    Text Length Range:
    • Min: {min(stats['text_lengths'])} chars
    • Max: {max(stats['text_lengths'])} chars
    
    Word Count Range:
    • Min: {min(stats['word_counts'])} words
    • Max: {max(stats['word_counts'])} words
    """
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 词频分析
    plt.figure(figsize=(15, 10))

    # 最常见词汇
    top_words = dict(list(filtered_word_freq.most_common(30)))
    plt.subplot(2, 2, 1)
    words, counts = zip(*top_words.items())
    plt.barh(range(len(words)), counts, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title('Top 30 Most Frequent Words')
    plt.gca().invert_yaxis()

    # 词频分布
    plt.subplot(2, 2, 2)
    if HAS_WORDCLOUD:
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(filtered_word_freq)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
    else:
        # 替代方案：显示词频分布
        word_lengths = [len(word) for word in filtered_word_freq.keys()]
        plt.hist(word_lengths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.title('Word Length Distribution')

    # 遥感关键词分析
    plt.subplot(2, 2, 3)
    category_totals = {cat: sum(counts.values()) for cat, counts in keyword_counts.items()}
    categories, totals = zip(*category_totals.items())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    plt.pie(totals, labels=categories, autopct='%1.1f%%', colors=colors)
    plt.title('Remote Sensing Keywords by Category')

    # 图像尺寸分析
    if image_sizes:
        plt.subplot(2, 2, 4)
        widths, heights = zip(*image_sizes)
        plt.scatter(widths, heights, alpha=0.6, color='coral')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title(f'Image Dimensions (n={len(image_sizes)})')

        # 添加统计信息
        avg_width, avg_height = np.mean(widths), np.mean(heights)
        plt.axvline(avg_width, color='red', linestyle='--', alpha=0.7, label=f'Avg W: {avg_width:.0f}')
        plt.axhline(avg_height, color='red', linestyle='--', alpha=0.7, label=f'Avg H: {avg_height:.0f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/vocabulary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 详细关键词分析
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Remote Sensing Keywords Analysis', fontsize=16, fontweight='bold')

    categories = list(keyword_counts.keys())
    for i, category in enumerate(categories):
        if i < 6:  # 最多显示6个类别
            row, col = i // 3, i % 3
            top_keywords = dict(list(keyword_counts[category].most_common(10)))

            if top_keywords:
                keywords, counts = zip(*top_keywords.items())
                axes[row, col].barh(range(len(keywords)), counts, color=colors[i % len(colors)])
                axes[row, col].set_yticks(range(len(keywords)))
                axes[row, col].set_yticklabels(keywords)
                axes[row, col].set_xlabel('Frequency')
                axes[row, col].set_title(f'{category.title()} Keywords')
                axes[row, col].invert_yaxis()

    # 隐藏多余的子图
    for i in range(len(categories), 6):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/keyword_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 生成数据集报告
    generate_dataset_report(stats, word_freq, filtered_word_freq, keyword_counts, image_sizes, file_sizes, output_dir)

    return output_dir

def generate_dataset_report(stats, word_freq, filtered_word_freq, keyword_counts, image_sizes, file_sizes, output_dir):
    """Generate detailed dataset report"""

    report = f"""
# RSICap Dataset Analysis Report

## Dataset Overview
- **Total Samples**: {stats['total_samples']:,}
- **Average Caption Length**: {stats['avg_text_length']:.1f} characters
- **Average Word Count**: {stats['avg_word_count']:.1f} words
- **Average Sentence Count**: {stats['avg_sentence_count']:.1f} sentences

## Text Statistics
- **Character Length Range**: {min(stats['text_lengths'])} - {max(stats['text_lengths'])}
- **Word Count Range**: {min(stats['word_counts'])} - {max(stats['word_counts'])}
- **Sentence Count Range**: {min(stats['sentence_counts'])} - {max(stats['sentence_counts'])}

## Vocabulary Statistics
- **Total Unique Words**: {len(word_freq):,}
- **Filtered Vocabulary Size**: {len(filtered_word_freq):,} (excluding stop words)
- **Most Common Words**: {', '.join([word for word, _ in list(filtered_word_freq.most_common(10))])}

## Remote Sensing Keywords
"""

    for category, counts in keyword_counts.items():
        if counts:
            total_count = sum(counts.values())
            top_keywords = ', '.join([word for word, _ in list(counts.most_common(5))])
            report += f"- **{category.title()}**: {total_count} mentions, top words: {top_keywords}\n"

    if image_sizes:
        widths, heights = zip(*image_sizes)
        report += f"""
## Image Properties (Sample Analysis)
- **Sample Size**: {len(image_sizes)} images
- **Average Dimensions**: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels
- **Dimension Range**: {min(widths)}-{max(widths)} x {min(heights)}-{max(heights)} pixels
"""

    if file_sizes:
        report += f"""
- **Average File Size**: {np.mean(file_sizes):.2f} MB
- **File Size Range**: {min(file_sizes):.2f} - {max(file_sizes):.2f} MB
"""

    report += f"""
## Analysis Summary
This dataset contains {stats['total_samples']} remote sensing image-caption pairs with rich descriptions averaging {stats['avg_word_count']:.1f} words per caption. The vocabulary is diverse with {len(filtered_word_freq):,} unique meaningful words, heavily focused on remote sensing terminology including objects, terrain features, urban elements, and spatial relationships.

Generated by RSICap Dataset Analyzer
"""

    # Save report
    with open(f'{output_dir}/dataset_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Dataset report saved to {output_dir}/dataset_report.md")

def main():
    """Main function"""
    print("Starting RSICap Dataset Analysis...")

    # Load data
    annotations, images_dir = load_rsicap_data()
    print(f"Loaded {len(annotations)} annotations")

    # Text analysis
    print("Analyzing text statistics...")
    stats, captions = analyze_text_statistics(annotations)

    # Vocabulary analysis
    print("Analyzing vocabulary...")
    word_freq, filtered_word_freq = analyze_vocabulary(captions)

    # Keyword analysis
    print("Analyzing remote sensing keywords...")
    keyword_counts = analyze_remote_sensing_keywords(captions)

    # Image analysis
    print("Analyzing image properties...")
    image_sizes, file_sizes = analyze_image_properties(annotations, images_dir)

    # Create visualizations
    print("Creating visualizations...")
    output_dir = create_visualizations(stats, captions, word_freq, filtered_word_freq,
                                     keyword_counts, image_sizes, file_sizes)

    print(f"Analysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
