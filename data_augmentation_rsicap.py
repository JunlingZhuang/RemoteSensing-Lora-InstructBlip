#!/usr/bin/env python3
"""
RSICap Dataset Image Augmentation Script
Supports configurable augmentation techniques for remote sensing images.

USAGE COMMANDS:

1. Preview augmentation settings:
   python data_augmentation_rsicap.py --data_dir data/rsgpt_dataset/RSICap --output_dir data/rsicap_augmented --preview

2. Generate 2x augmented dataset (6,204 total images):
   python data_augmentation_rsicap.py --data_dir data/rsgpt_dataset/RSICap --output_dir data/rsicap_augmented --num_aug 2

3. Use custom configuration:
   python data_augmentation_rsicap.py --data_dir data/rsgpt_dataset/RSICap --output_dir data/rsicap_augmented --config configs/augmentation_config.yml --num_aug 3

4. Conservative augmentation (1 per image):
   python data_augmentation_rsicap.py --data_dir data/rsgpt_dataset/RSICap --output_dir data/rsicap_augmented --num_aug 1

OUTPUT:
- Augmented images in output_dir/images/
- Updated annotations.json with all image-caption pairs
- augmentation_config.yaml for reproducibility

AUGMENTATION FACTOR:
- Default config: ~7x theoretical factor
- With --num_aug 2: 3x actual dataset size (original + 2 augmented per image)
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import random
from tqdm import tqdm
import yaml


class RSICapAugmentation:
    """Configurable augmentation for RSICap remote sensing images."""
    
    def __init__(self, config):
        """
        Initialize augmentation with configuration.
        
        Args:
            config: Dictionary with augmentation settings
        """
        self.config = config
        self.aug_prob = config.get('aug_prob', 0.8)
        
        # Build augmentation pipeline based on config
        self.transforms = self._build_transforms()
        
    def _build_transforms(self):
        """Build augmentation transforms based on configuration."""
        transform_list = []
        
        # 1. Spatial Transforms (保持遥感图像的方位特性)
        if self.config.get('enable_spatial', True):
            spatial_transforms = []
            
            # Random Resized Crop (模拟不同拍摄高度)
            if self.config.get('enable_crop', True):
                spatial_transforms.append(
                    transforms.RandomResizedCrop(
                        size=self.config.get('image_size', 512),  # 默认保持原始大小
                        scale=tuple(self.config.get('crop_scale', [0.8, 1.0])),
                        ratio=tuple(self.config.get('crop_ratio', [0.9, 1.1]))
                    )
                )
            
            # Random Affine (小范围平移，不旋转)
            if self.config.get('enable_affine', True):
                spatial_transforms.append(
                    transforms.RandomAffine(
                        degrees=0,  # 不旋转，保持方位
                        translate=tuple(self.config.get('translate_range', [0.1, 0.1])),
                        scale=None,
                        shear=0
                    )
                )
            
            if spatial_transforms:
                transform_list.extend(spatial_transforms)
        
        # 2. Color Transforms (不影响空间信息)
        if self.config.get('enable_color', True):
            color_transforms = []
            
            # Color Jitter (模拟不同光照和大气条件)
            if self.config.get('enable_color_jitter', True):
                color_transforms.append(
                    transforms.ColorJitter(
                        brightness=self.config.get('brightness', 0.2),
                        contrast=self.config.get('contrast', 0.2),
                        saturation=self.config.get('saturation', 0.1),
                        hue=self.config.get('hue', 0.05)
                    )
                )
            
            # Random Grayscale (模拟单波段图像)
            if self.config.get('enable_grayscale', True):
                color_transforms.append(
                    transforms.RandomGrayscale(
                        p=self.config.get('grayscale_prob', 0.1)
                    )
                )
            
            # Gaussian Blur (模拟大气散射和传感器模糊)
            if self.config.get('enable_blur', True):
                color_transforms.append(
                    transforms.GaussianBlur(
                        kernel_size=self.config.get('blur_kernel', 3),
                        sigma=tuple(self.config.get('blur_sigma', [0.1, 0.5]))
                    )
                )
            
            if color_transforms:
                transform_list.extend(color_transforms)
        
        # 3. Noise and Quality Transforms
        if self.config.get('enable_noise', False):
            # Add custom noise transforms here if needed
            pass
        
        # 4. Ensure tensor conversion
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)
    
    def augment_image(self, image_path):
        """
        Apply augmentation to a single image.

        Args:
            image_path: Path to input image

        Returns:
            Augmented image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)

            # Create dynamic transforms based on original image size
            dynamic_transforms = self._build_dynamic_transforms(original_size)

            # Apply transforms with probability
            if random.random() < self.aug_prob:
                augmented = dynamic_transforms(image)
            else:
                # Just resize to original size and convert to tensor
                resize_transform = transforms.Compose([
                    transforms.Resize(original_size),
                    transforms.ToTensor()
                ])
                augmented = resize_transform(image)

            return augmented

        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return None

    def _build_dynamic_transforms(self, original_size):
        """Build transforms that preserve original image size.

        Args:
            original_size: Tuple (width, height) of original image

        Returns:
            Composed transforms
        """
        transform_list = []

        # Use the larger dimension to ensure we don't lose information
        target_size = max(original_size)

        # 1. Spatial Transforms (保持遥感图像的方位特性)
        if self.config.get('enable_spatial', True):
            spatial_transforms = []

            # Random Resized Crop (模拟不同拍摄高度)
            if self.config.get('enable_crop', True):
                spatial_transforms.append(
                    transforms.RandomResizedCrop(
                        size=target_size,
                        scale=tuple(self.config.get('crop_scale', [0.8, 1.0])),
                        ratio=tuple(self.config.get('crop_ratio', [0.9, 1.1]))
                    )
                )

            # Random Affine (小范围平移，不旋转)
            if self.config.get('enable_affine', True):
                spatial_transforms.append(
                    transforms.RandomAffine(
                        degrees=0,  # 不旋转，保持方位
                        translate=tuple(self.config.get('translate_range', [0.1, 0.1])),
                        scale=None,
                        shear=0
                    )
                )

            if spatial_transforms:
                transform_list.extend(spatial_transforms)

        # 2. Color Transforms (保持遥感图像的光谱特性)
        if self.config.get('enable_color', True):
            color_transforms = []

            # Color Jitter (模拟不同光照条件)
            if self.config.get('enable_color_jitter', True):
                color_transforms.append(
                    transforms.ColorJitter(
                        brightness=self.config.get('brightness', 0.2),
                        contrast=self.config.get('contrast', 0.2),
                        saturation=self.config.get('saturation', 0.1),
                        hue=self.config.get('hue', 0.05)
                    )
                )

            # Random Grayscale (模拟单波段图像)
            if self.config.get('enable_grayscale', True):
                color_transforms.append(
                    transforms.RandomGrayscale(
                        p=self.config.get('grayscale_prob', 0.1)
                    )
                )

            if color_transforms:
                transform_list.extend(color_transforms)

        # 3. Noise and Blur (模拟传感器噪声和大气影响)
        if self.config.get('enable_blur', True):
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=self.config.get('blur_kernel', 3),
                        sigma=tuple(self.config.get('blur_sigma', [0.1, 2.0]))
                    )
                ], p=self.config.get('blur_prob', 0.2))
            )

        # Final resize to original size and convert to tensor
        transform_list.extend([
            transforms.Resize(original_size),  # Resize back to original size
            transforms.ToTensor()
        ])

        return transforms.Compose(transform_list)
    
    def get_augmentation_factor(self):
        """Calculate theoretical augmentation factor."""
        factor = 1.0
        
        # Spatial augmentations
        if self.config.get('enable_spatial', True):
            spatial_factor = 1.0
            if self.config.get('enable_crop', True):
                spatial_factor *= 2.0  # Different crops
            if self.config.get('enable_affine', True):
                spatial_factor *= 1.5  # Different translations
            factor *= spatial_factor
        
        # Color augmentations
        if self.config.get('enable_color', True):
            color_factor = 1.0
            if self.config.get('enable_color_jitter', True):
                color_factor *= 3.0  # Different color variations
            if self.config.get('enable_grayscale', True):
                color_factor *= 1.1  # 10% grayscale chance
            if self.config.get('enable_blur', True):
                color_factor *= 1.2  # Different blur levels
            factor *= color_factor
        
        # Apply augmentation probability
        factor *= self.aug_prob
        
        return factor


def load_rsicap_annotations(data_dir):
    """Load RSICap annotations."""
    # RSICap uses 'captions.json' not 'annotations.json'
    annotations_path = os.path.join(data_dir, 'captions.json')

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Captions not found: {annotations_path}")

    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # RSICap format has annotations under 'annotations' key
    if 'annotations' in data:
        annotations = data['annotations']
    else:
        annotations = data

    return annotations


def augment_rsicap_dataset(data_dir, output_dir, config, num_augmentations=1):
    """
    Augment RSICap dataset images.
    
    Args:
        data_dir: Path to original RSICap dataset
        output_dir: Path to save augmented images
        config: Augmentation configuration
        num_augmentations: Number of augmented versions per image
    """
    print(f"Starting RSICap dataset augmentation...")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    annotations = load_rsicap_annotations(data_dir)
    print(f"Found {len(annotations)} image annotations")

    # Initialize augmentation
    augmenter = RSICapAugmentation(config)
    theoretical_factor = augmenter.get_augmentation_factor()

    print(f"Theoretical augmentation factor: {theoretical_factor:.1f}x")
    print(f"Requested augmentations per image: {num_augmentations}")
    
    # Process images
    images_dir = os.path.join(data_dir, 'images')
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    augmented_annotations = []
    total_generated = 0
    current_image_id = 0

    for annotation in tqdm(annotations, desc="Augmenting images"):
        image_filename = annotation['filename']
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue

        # Copy original image to output directory
        original_output_path = os.path.join(output_images_dir, image_filename)
        import shutil
        shutil.copy2(image_path, original_output_path)

        # Add original annotation with new image_id
        original_annotation = annotation.copy()
        original_annotation['image_id'] = str(current_image_id)
        augmented_annotations.append(original_annotation)
        current_image_id += 1
        
        # Generate augmented versions
        for aug_idx in range(num_augmentations):
            try:
                # Generate augmented image
                augmented_tensor = augmenter.augment_image(image_path)
                
                if augmented_tensor is not None:
                    # Convert back to PIL and save
                    augmented_pil = transforms.ToPILImage()(augmented_tensor)
                    
                    # Create new filename
                    name, ext = os.path.splitext(image_filename)
                    aug_filename = f"{name}_aug{aug_idx+1}{ext}"
                    aug_path = os.path.join(output_images_dir, aug_filename)
                    
                    # Save augmented image
                    augmented_pil.save(aug_path)
                    
                    # Create new annotation with unique image_id
                    aug_annotation = annotation.copy()
                    aug_annotation['image_id'] = str(current_image_id)
                    aug_annotation['filename'] = aug_filename
                    aug_annotation['original_filename'] = image_filename
                    aug_annotation['augmentation_id'] = aug_idx + 1

                    augmented_annotations.append(aug_annotation)
                    current_image_id += 1
                    total_generated += 1
                    
            except Exception as e:
                print(f"❌ Error augmenting {image_filename} (aug {aug_idx+1}): {e}")
    
    # Save augmented annotations in RSICap format
    output_annotations_path = os.path.join(output_dir, 'captions.json')
    output_data = {"annotations": augmented_annotations}
    with open(output_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    # Save augmentation config
    config_path = os.path.join(output_dir, 'augmentation_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nAugmentation completed!")
    print(f"Original images: {len(annotations)}")
    print(f"Generated augmented images: {total_generated}")
    print(f"Total images: {len(augmented_annotations)}")
    print(f"Actual augmentation factor: {len(augmented_annotations)/len(annotations):.1f}x")
    print(f"Augmented dataset saved to: {output_dir}")


def create_default_config():
    """Create default augmentation configuration."""
    return {
        # Global settings
        'aug_prob': 0.8,
        'image_size': 224,
        
        # Spatial transforms
        'enable_spatial': True,
        'enable_crop': True,
        'crop_scale': [0.8, 1.0],
        'crop_ratio': [0.9, 1.1],
        'enable_affine': True,
        'translate_range': [0.1, 0.1],
        
        # Color transforms
        'enable_color': True,
        'enable_color_jitter': True,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05,
        'enable_grayscale': True,
        'grayscale_prob': 0.1,
        'enable_blur': True,
        'blur_kernel': 3,
        'blur_sigma': [0.1, 0.5],
        
        # Noise transforms
        'enable_noise': False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSICap Dataset Augmentation")
    parser.add_argument("--data_dir", required=True, help="Path to RSICap dataset")
    parser.add_argument("--output_dir", required=True, help="Path to save augmented dataset")
    parser.add_argument("--config", help="Path to augmentation config YAML file")
    parser.add_argument("--num_aug", type=int, default=1, help="Number of augmentations per image")
    parser.add_argument("--preview", action="store_true", help="Preview augmentation settings only")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")

    if args.preview:
        # Preview mode
        augmenter = RSICapAugmentation(config)
        factor = augmenter.get_augmentation_factor()

        print(f"\nAugmentation Preview:")
        print(f"   Theoretical factor: {factor:.1f}x")
        print(f"   Requested augmentations: {args.num_aug}")
        print(f"   Expected total factor: {(1 + args.num_aug):.1f}x")
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    else:
        # Run augmentation
        augment_rsicap_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            num_augmentations=args.num_aug
        )
