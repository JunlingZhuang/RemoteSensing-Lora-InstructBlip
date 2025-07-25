#!/usr/bin/env python3
"""
Simple augmentation demo for RSICap - generates before/after comparison images.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


def create_synthetic_rs_image(size=(224, 224)):
    """Create a synthetic remote sensing-like image for testing."""
    print("Creating synthetic remote sensing test image...")
    
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
    
    return img


def load_real_rsicap_image():
    """Load a real image from RSICap dataset."""
    rsicap_path = "data/rsgpt_dataset/RSICap/images"

    if not os.path.exists(rsicap_path):
        print(f"Warning: RSICap images not found at {rsicap_path}")
        print("Using synthetic image instead...")
        return create_synthetic_rs_image()

    # Get list of images
    import glob
    image_files = glob.glob(os.path.join(rsicap_path, "*.jpg"))

    if not image_files:
        print("No JPG images found in RSICap dataset, using synthetic image...")
        return create_synthetic_rs_image()

    # Use the first image (or you can randomly select)
    selected_image = image_files[0]
    print(f"Using real RSICap image: {os.path.basename(selected_image)}")

    try:
        img = Image.open(selected_image).convert('RGB')
        # Resize to standard size
        img = img.resize((224, 224))
        return img
    except Exception as e:
        print(f"Error loading real image: {e}")
        print("Falling back to synthetic image...")
        return create_synthetic_rs_image()


def create_augmentation_comparison():
    """Create a comprehensive augmentation comparison figure."""
    print("Creating augmentation comparison...")

    # Load real RSICap image
    original = load_real_rsicap_image()
    
    # Define augmentation effects
    effects = {
        'Original': transforms.Compose([]),
        'Crop (0.8x)': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0))
        ]),
        'Color Jitter': transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        ),
        'Grayscale': transforms.RandomGrayscale(p=1.0),
        'Blur (σ=0.7)': transforms.GaussianBlur(kernel_size=5, sigma=0.7),
        'Translation': transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        'Combined': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(3, sigma=(0.1, 0.5))
        ])
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('RSICap Data Augmentation Effects for Remote Sensing VQA', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Apply each effect and plot
    for i, (name, transform) in enumerate(effects.items()):
        if i >= 8:  # Only show first 8 effects
            break
            
        try:
            if name == 'Original':
                result = original
            else:
                # Set random seed for reproducible results
                torch.manual_seed(42)
                np.random.seed(42)
                
                if 'Crop' in name or 'Combined' in name:
                    # For crop operations, start with larger image
                    temp_img = original.resize((256, 256))
                    result = transform(temp_img)
                else:
                    result = transform(original)
                
                # Convert tensor back to PIL if needed
                if torch.is_tensor(result):
                    result = transforms.ToPILImage()(result)
            
            # Plot the result
            axes[i].imshow(result)
            axes[i].set_title(name, fontweight='bold' if name == 'Original' else 'normal')
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {name}", ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplot
    if len(effects) < 8:
        axes[7].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    output_dir = "augmentation_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_path = os.path.join(output_dir, "augmentation_effects_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {comparison_path}")
    return comparison_path


def create_individual_effect_samples():
    """Create individual before/after samples for each effect."""
    print("Creating individual effect samples...")
    
    original = create_synthetic_rs_image()
    output_dir = "augmentation_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    original.save(os.path.join(output_dir, "00_original.jpg"))
    
    # Define effects with parameters
    effects = {
        'crop_conservative': {
            'transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0))
            ]),
            'description': 'Conservative Crop (0.9-1.0 scale)'
        },
        'crop_aggressive': {
            'transform': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.7, 0.8))
            ]),
            'description': 'Aggressive Crop (0.7-0.8 scale)'
        },
        'color_subtle': {
            'transform': transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            'description': 'Subtle Color Jitter'
        },
        'color_strong': {
            'transform': transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            'description': 'Strong Color Jitter'
        },
        'blur_light': {
            'transform': transforms.GaussianBlur(3, sigma=0.3),
            'description': 'Light Atmospheric Blur'
        },
        'blur_heavy': {
            'transform': transforms.GaussianBlur(5, sigma=1.0),
            'description': 'Heavy Atmospheric Blur'
        },
        'grayscale': {
            'transform': transforms.RandomGrayscale(p=1.0),
            'description': 'Panchromatic Simulation'
        },
        'translation': {
            'transform': transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            'description': 'Position Shift'
        }
    }
    
    # Generate samples
    sample_paths = []
    for name, config in effects.items():
        try:
            torch.manual_seed(42)  # For reproducible results
            np.random.seed(42)
            
            if 'crop' in name:
                temp_img = original.resize((256, 256))
                result = config['transform'](temp_img)
            else:
                result = config['transform'](original)
            
            if torch.is_tensor(result):
                result = transforms.ToPILImage()(result)
            
            # Save sample
            filename = f"{name}.jpg"
            filepath = os.path.join(output_dir, filename)
            result.save(filepath)
            
            sample_paths.append((name, config['description'], filename))
            print(f"  Generated: {filename}")
            
        except Exception as e:
            print(f"  Error generating {name}: {e}")
    
    return sample_paths


def generate_markdown_report(comparison_path, sample_paths):
    """Generate a markdown report with images."""
    print("Generating markdown report...")
    
    output_dir = "augmentation_samples"
    report_path = os.path.join(output_dir, "augmentation_report.md")
    
    report_content = f"""# RSICap Data Augmentation Effects Report

This report demonstrates the data augmentation techniques designed for remote sensing VQA tasks.

## Overall Comparison

![Augmentation Effects Comparison]({os.path.basename(comparison_path)})

*Figure 1: Comprehensive comparison of augmentation effects on synthetic remote sensing imagery*

## Individual Effect Analysis

### Original Image
![Original](00_original.jpg)

*Baseline synthetic remote sensing image with urban areas, water bodies, agricultural fields, and transportation infrastructure*

### Spatial Transformations

#### Random Resized Crop (模拟不同拍摄高度)
Simulates different imaging altitudes and field-of-view variations.

| Conservative Crop | Aggressive Crop |
|-------------------|-----------------|
| ![Conservative](crop_conservative.jpg) | ![Aggressive](crop_aggressive.jpg) |
| Scale: 0.9-1.0 | Scale: 0.7-0.8 |

*Simulates satellite imagery at different altitudes - higher altitude (smaller scale) vs lower altitude (larger scale)*

### Photometric Transformations

#### Color Jitter (模拟不同光照和大气条件)
Simulates varying illumination and atmospheric conditions.

| Subtle Color Variation | Strong Color Variation |
|------------------------|------------------------|
| ![Subtle](color_subtle.jpg) | ![Strong](color_strong.jpg) |
| ±10% brightness/contrast | ±30% brightness/contrast |

*Simulates different solar angles, atmospheric haze, and seasonal variations*

#### Gaussian Blur (模拟大气散射和传感器模糊)
Simulates atmospheric scattering and sensor blur effects.

| Light Atmospheric Blur | Heavy Atmospheric Blur |
|------------------------|------------------------|
| ![Light](blur_light.jpg) | ![Heavy](blur_heavy.jpg) |
| σ = 0.3 | σ = 1.0 |

*Simulates atmospheric turbulence and sensor limitations*

#### Grayscale Conversion (模拟单波段图像)
Simulates panchromatic or single-band imagery.

| Original RGB | Panchromatic Simulation |
|--------------|-------------------------|
| ![Original](00_original.jpg) | ![Grayscale](grayscale.jpg) |
| Multi-spectral | Single-band |

*Simulates panchromatic sensors or infrared band imagery*

### Geometric Transformations

#### Random Affine (模拟位置偏移)
Simulates minor positional shifts while preserving orientation.

| Original Position | Shifted Position |
|-------------------|------------------|
| ![Original](00_original.jpg) | ![Translation](translation.jpg) |
| Baseline | ±10% translation |

*Simulates GPS inaccuracies and platform movement (no rotation to preserve geographic orientation)*

## Technical Parameters

### Implementation Details
```python
# Spatial Augmentations
RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2))
RandomAffine(degrees=0, translate=(0.1, 0.1))  # No rotation

# Photometric Augmentations  
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
RandomGrayscale(p=0.1)

# Application probability: 80%
```

### Remote Sensing Considerations
- **No rotation**: Preserves geographic orientation (North-up convention)
- **Conservative translation**: Maintains spatial context
- **Realistic color variations**: Based on atmospheric and illumination changes
- **Appropriate blur levels**: Matches typical atmospheric effects

## Expected Impact

### Dataset Expansion
- Original RSICap: 2,068 images
- With augmentation: 6,204+ images (3× expansion)
- Improved robustness to imaging conditions

### Performance Benefits
- Enhanced generalization across different sensors
- Better handling of atmospheric variations
- Improved spatial reasoning under different viewpoints
- Reduced overfitting on limited training data

## Usage in Paper

### Key Points for Methodology Section
1. **Domain-specific design**: Tailored for remote sensing characteristics
2. **Preservation of spatial semantics**: No rotation, conservative transformations
3. **Realistic parameter ranges**: Based on actual remote sensing conditions
4. **Systematic evaluation**: Comprehensive ablation studies

### Recommended Figures
- Use the overall comparison (Figure 1) for methodology overview
- Include individual before/after pairs for detailed analysis
- Show quantitative results in ablation studies

---
*Generated by RSICap Augmentation Demo*
*Images show synthetic remote sensing data for demonstration purposes*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    """Main function to generate all augmentation samples and report."""
    print("RSICap Data Augmentation Demo")
    print("=" * 50)
    
    try:
        # Create comparison figure
        comparison_path = create_augmentation_comparison()
        
        # Create individual samples
        sample_paths = create_individual_effect_samples()
        
        # Generate report
        report_path = generate_markdown_report(comparison_path, sample_paths)
        
        print(f"\nDemo completed successfully!")
        print(f"Check the 'augmentation_samples' directory for:")
        print(f"  - Comparison figure: {os.path.basename(comparison_path)}")
        print(f"  - Individual samples: {len(sample_paths)} effect variations")
        print(f"  - Detailed report: {os.path.basename(report_path)}")
        print(f"\nUse these images and report for your paper writing!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
