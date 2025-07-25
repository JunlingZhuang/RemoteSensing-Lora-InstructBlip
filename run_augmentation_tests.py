#!/usr/bin/env python3
"""
Quick test runner for augmentation effects visualization.
Creates sample images for paper writing without needing the full RSICap dataset.

USAGE:
python run_augmentation_tests.py

This script will:
1. Create a synthetic remote sensing-like test image
2. Run all augmentation effect tests
3. Generate comparison grids and documentation
4. Provide ready-to-use figures for paper writing
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import sys


def create_synthetic_rs_image(output_path, size=(224, 224)):
    """Create a synthetic remote sensing-like image for testing."""
    print("🎨 Creating synthetic remote sensing test image...")
    
    # Create base image with different regions
    img = Image.new('RGB', size, color=(34, 139, 34))  # Forest green base
    draw = ImageDraw.Draw(img)
    
    # Add urban area (gray)
    urban_coords = [50, 50, 120, 120]
    draw.rectangle(urban_coords, fill=(128, 128, 128))
    
    # Add buildings (darker gray)
    for i in range(3):
        for j in range(3):
            x1 = 55 + i * 20
            y1 = 55 + j * 20
            x2 = x1 + 15
            y2 = y1 + 15
            draw.rectangle([x1, y1, x2, y2], fill=(64, 64, 64))
    
    # Add water body (blue)
    water_coords = [140, 30, 200, 90]
    draw.ellipse(water_coords, fill=(30, 144, 255))
    
    # Add agricultural fields (different greens and browns)
    field_colors = [(154, 205, 50), (107, 142, 35), (160, 82, 45), (210, 180, 140)]
    for i, color in enumerate(field_colors):
        x1 = 30 + (i % 2) * 80
        y1 = 140 + (i // 2) * 40
        x2 = x1 + 70
        y2 = y1 + 35
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Add roads (dark gray lines)
    draw.line([0, 100, 224, 100], fill=(64, 64, 64), width=3)
    draw.line([100, 0, 100, 224], fill=(64, 64, 64), width=3)
    
    # Add some texture/noise to make it more realistic
    pixels = np.array(img)
    noise = np.random.normal(0, 5, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    img.save(output_path)
    print(f"✅ Synthetic test image saved to: {output_path}")
    return output_path


def run_augmentation_tests():
    """Run all augmentation effect tests."""
    print("🚀 Running RSICap Augmentation Effect Tests")
    print("=" * 60)
    
    # Create test image
    test_image_path = "test_rs_image.jpg"
    create_synthetic_rs_image(test_image_path)
    
    # Create output directory
    output_dir = "augmentation_effect_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test individual effects
    effects = ['crop', 'color', 'grayscale', 'blur', 'affine']
    
    print(f"\n🧪 Testing individual effects...")
    for effect in effects:
        print(f"   Testing {effect} effect...")
        try:
            cmd = [
                sys.executable, "test_augmentation_effects.py",
                "--input_image", test_image_path,
                "--output_dir", output_dir,
                "--effect", effect
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ {effect} effect completed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {effect} effect failed: {e}")
            print(f"      Stdout: {e.stdout}")
            print(f"      Stderr: {e.stderr}")
    
    # Generate comparison grid
    print(f"\n🎨 Generating comparison grid...")
    try:
        cmd = [
            sys.executable, "test_augmentation_effects.py",
            "--input_image", test_image_path,
            "--output_dir", output_dir,
            "--grid"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ Comparison grid completed")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Comparison grid failed: {e}")
    
    # Generate summary report
    generate_summary_report(output_dir)
    
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print(f"\n🎉 All tests completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Check the comparison grid for paper figures")
    print(f"📋 Check summary_report.md for detailed documentation")


def generate_summary_report(output_dir):
    """Generate a summary report for paper writing."""
    print("Generating summary report...")

    # Check which effects were successfully generated
    effects_generated = []
    for effect in ['crop', 'color', 'grayscale', 'blur', 'affine']:
        effect_dir = os.path.join(output_dir, effect)
        if os.path.exists(effect_dir):
            effects_generated.append(effect)

    # Check if comparison grid exists
    grid_exists = os.path.exists(os.path.join(output_dir, 'augmentation_comparison_grid.png'))

    report_content = f"""# RSICap Data Augmentation Effects Summary

This report documents the augmentation effects tested for the RSICap dataset, designed for remote sensing VQA tasks.

## Generated Files Overview

- **Comparison Grid**: {'✅ Generated' if grid_exists else '❌ Not generated'} - `augmentation_comparison_grid.png`
- **Individual Effects**: {len(effects_generated)}/5 effects generated
  - Generated: {', '.join(effects_generated) if effects_generated else 'None'}

## Visual Examples

### Overall Comparison
{'![Augmentation Comparison Grid](augmentation_comparison_grid.png)' if grid_exists else '*Comparison grid not available*'}

*Figure: Comprehensive comparison of all augmentation effects applied to a synthetic remote sensing image*

## Augmentation Techniques

### 1. Random Resized Crop (模拟不同拍摄高度)
- **Purpose**: Simulates different imaging altitudes and field-of-view variations
- **Parameters**:
  - Scale range: [0.7, 1.0]
  - Aspect ratio: [0.8, 1.2]
- **Variants**: Conservative (0.9-1.0), Moderate (0.8-1.0), Aggressive (0.7-1.0)
- **Remote Sensing Relevance**: Different satellite/aircraft altitudes, zoom levels

{'#### Visual Examples' if 'crop' in effects_generated else '#### Examples: Not Generated'}
{'![Original](crop/00_original.jpg) ![Conservative](crop/01_conservative_sample1.jpg) ![Moderate](crop/02_moderate_sample1.jpg) ![Aggressive](crop/03_aggressive_sample1.jpg)' if 'crop' in effects_generated else '*Crop effect samples not available*'}

*Before/After: Original → Conservative → Moderate → Aggressive cropping*

### 2. Color Jitter (模拟不同光照和大气条件)
- **Purpose**: Simulates varying illumination and atmospheric conditions
- **Parameters**:
  - Brightness: ±30%
  - Contrast: ±30%
  - Saturation: ±20%
  - Hue: ±10%
- **Variants**: Subtle (±10%), Moderate (±20%), Strong (±30%)
- **Remote Sensing Relevance**: Solar angle variations, atmospheric haze, seasonal changes

{'#### Visual Examples' if 'color' in effects_generated else '#### Examples: Not Generated'}
{'![Original](color/00_original.jpg) ![Subtle](color/01_subtle_sample1.jpg) ![Moderate](color/02_moderate_sample1.jpg) ![Strong](color/03_strong_sample1.jpg)' if 'color' in effects_generated else '*Color jitter effect samples not available*'}

*Before/After: Original → Subtle → Moderate → Strong color variations*

### 3. Random Grayscale (模拟单波段图像)
- **Purpose**: Simulates single-band or panchromatic imagery
- **Parameters**: Probability of conversion (10-20% typical)
- **Remote Sensing Relevance**: Panchromatic sensors, infrared bands, historical imagery

{'#### Visual Examples' if 'grayscale' in effects_generated else '#### Examples: Not Generated'}
{'![Original](grayscale/00_original.jpg) ![Grayscale](grayscale/03_always_demo_sample1.jpg)' if 'grayscale' in effects_generated else '*Grayscale effect samples not available*'}

*Before/After: Original RGB → Grayscale conversion (simulating panchromatic imagery)*

### 4. Gaussian Blur (模拟大气散射和传感器模糊)
- **Purpose**: Simulates atmospheric scattering and sensor blur effects
- **Parameters**:
  - Kernel size: 3-5 pixels
  - Sigma range: [0.1, 1.2]
- **Variants**: Light (σ=0.1-0.3), Medium (σ=0.3-0.7), Heavy (σ=0.7-1.2)
- **Remote Sensing Relevance**: Atmospheric turbulence, sensor limitations, motion blur

{'#### Visual Examples' if 'blur' in effects_generated else '#### Examples: Not Generated'}
{'![Original](blur/00_original.jpg) ![Light](blur/01_light_blur_sample1.jpg) ![Medium](blur/02_medium_blur_sample1.jpg) ![Heavy](blur/03_heavy_blur_sample1.jpg)' if 'blur' in effects_generated else '*Blur effect samples not available*'}

*Before/After: Original → Light → Medium → Heavy atmospheric blur*

### 5. Random Affine (模拟位置偏移)
- **Purpose**: Simulates minor positional shifts while preserving orientation
- **Parameters**:
  - Translation: ±5% to ±15% of image size
  - No rotation (preserves geographic orientation)
- **Remote Sensing Relevance**: GPS inaccuracies, registration errors, platform movement

{'#### Visual Examples' if 'affine' in effects_generated else '#### Examples: Not Generated'}
{'![Original](affine/00_original.jpg) ![Small](affine/01_small_shift_sample1.jpg) ![Medium](affine/02_medium_shift_sample1.jpg) ![Large](affine/03_large_shift_sample1.jpg)' if 'affine' in effects_generated else '*Affine transformation samples not available*'}

*Before/After: Original → Small → Medium → Large positional shifts*

## Implementation Details

### Augmentation Pipeline
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])
```

### Application Probability
- Global augmentation probability: 80%
- Individual effect probabilities vary by technique
- Preserves 20% of original images for baseline comparison

## Expected Performance Impact

### Dataset Expansion
- Original RSICap: 2,068 images
- With 2 augmentations per image: 6,204 images (3× expansion)
- Theoretical augmentation factor: ~7× with full pipeline

### Performance Improvements (Expected)
- Improved generalization to different imaging conditions
- Better robustness to atmospheric variations
- Enhanced spatial reasoning under different viewpoints
- Reduced overfitting on limited training data

## Usage in Paper

### Methodology Section
- Emphasize domain-specific design for remote sensing
- Highlight preservation of geographic orientation
- Document parameter selection rationale

### Experimental Section
- Include ablation study comparing different augmentation strategies
- Show quantitative improvements in VQA accuracy
- Demonstrate robustness across different question types

### Figures
- Use comparison grid to show augmentation effects
- Include before/after examples for each technique
- Show parameter sensitivity analysis

## Files Generated
- Individual effect samples in subdirectories
- Comparison grid: `augmentation_comparison_grid.png`
- Parameter documentation: `effect_documentation.json` (per effect)
- This summary report: `summary_report.md`

---
Generated by RSICap Augmentation Effect Tester
"""
    
    report_path = os.path.join(output_dir, "summary_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Summary report saved to: {report_path}")


if __name__ == "__main__":
    try:
        run_augmentation_tests()
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
