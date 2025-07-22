"""
Unified inference interface for LoRA and native models.
Supports both fine-tuned LoRA models and original BLIP-2/InstructBLIP models.
"""

import os
import torch
from PIL import Image
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from peft import PeftModel
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config


class ModelInferencer:
    """Unified inferencer for different model types"""
    
    def __init__(self, model_type="instructblip", model_path=None, config=None):
        """
        Initialize inferencer
        
        Args:
            model_type: "instructblip", "blip2", or "lora"
            model_path: Path to model (for LoRA models) or None for native models
            config: Config object (only used for model loading, not generation)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.config = config or Config()
        
        # Default generation settings (from successful debug_instructblip.ipynb)
        self.generation_config = {
            "max_new_tokens": 300,
            "num_beams": 1,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "temperature": 1.0
        }
        
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on type"""
        print(f"Loading {self.model_type} model...")
        
        if self.model_type == "lora":
            self._load_lora_model()
        elif self.model_type == "instructblip":
            self._load_instructblip_model()
        elif self.model_type == "blip2":
            self._load_blip2_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_lora_model(self):
        """Load LoRA fine-tuned model"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"LoRA model path not found: {self.model_path}")
        
        # Load base InstructBLIP model
        print("Loading base InstructBLIP model...")
        base_model = InstructBlipForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from {self.model_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Load processor
        self.processor = InstructBlipProcessor.from_pretrained(self.config.model_name)
        
        print("LoRA model loaded successfully!")
    
    def _load_instructblip_model(self):
        """Load native InstructBLIP model"""
        model_name = self.config.model_name
        
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.processor = InstructBlipProcessor.from_pretrained(model_name)
         
        print("Native InstructBLIP model loaded successfully!")
    
    def _load_blip2_model(self):
        """Load native BLIP-2 model"""
        model_name = "Salesforce/blip2-flan-t5-xl"
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        print("Native BLIP-2 model loaded successfully!")
    
    def generate_caption(self, image_path, instruction=None, **generation_kwargs):
        """
        Generate caption for an image
        
        Args:
            image_path: Path to image file
            instruction: Custom instruction (optional)
            **generation_kwargs: Override default generation parameters
            
        Returns:
            Generated caption text
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare instruction
        if instruction is None:
            if self.model_type in ["instructblip", "lora"]:
                instruction = "Describe this remote sensing image in detail."
            else:  # blip2
                instruction = "a photo of"
        
        # Process inputs
        if self.model_type == "blip2":
            # BLIP-2 uses different input format
            inputs = self.processor(images=image, text=instruction, return_tensors="pt")
        else:
            # InstructBLIP format
            inputs = self.processor(images=image, text=instruction, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Merge default generation config with any overrides
        generation_params = self.generation_config.copy()
        generation_params.update(generation_kwargs)
        
        # Generate caption with torch.no_grad (from successful debug notebook)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )
        
        # Decode output
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up caption (remove instruction prompt)
        if instruction in caption:
            caption = caption.replace(instruction, "").strip()
        
        return caption
    
    def update_generation_config(self, **kwargs):
        """Update generation parameters"""
        self.generation_config.update(kwargs)
        print(f"Updated generation config: {kwargs}")
    
    def batch_inference(self, image_paths, instructions=None):
        """
        Generate captions for multiple images
        
        Args:
            image_paths: List of image file paths
            instructions: List of instructions (optional)
            
        Returns:
            List of generated captions
        """
        if instructions is None:
            instructions = [None] * len(image_paths)
        
        results = []
        for img_path, instruction in zip(image_paths, instructions):
            try:
                caption = self.generate_caption(img_path, instruction)
                results.append({
                    'image_path': img_path,
                    'instruction': instruction,
                    'caption': caption,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'instruction': instruction,
                    'caption': None,
                    'error': str(e),
                    'success': False
                })
                print(f"Error processing {img_path}: {e}")
        
        return results
    
    def compare_models(self, image_path, other_inferencer, instruction=None):
        """
        Compare outputs from two different models
        
        Args:
            image_path: Path to image
            other_inferencer: Another ModelInferencer instance
            instruction: Instruction text
            
        Returns:
            Comparison results
        """
        caption1 = self.generate_caption(image_path, instruction)
        caption2 = other_inferencer.generate_caption(image_path, instruction)
        
        return {
            'image_path': image_path,
            'instruction': instruction,
            f'{self.model_type}_caption': caption1,
            f'{other_inferencer.model_type}_caption': caption2,
            'model1_type': self.model_type,
            'model2_type': other_inferencer.model_type
        }


class InferenceDemo:
    """Demo class for showcasing inference capabilities"""
    
    def __init__(self):
        self.inferencers = {}
    
    def setup_models(self, lora_model_path=None):
        """Setup different model types for comparison"""
        print("Setting up models for comparison...")
        
        # Native InstructBLIP
        try:
            self.inferencers['instructblip'] = ModelInferencer(
                model_type="instructblip"
            )
            print("Native InstructBLIP ready")
        except Exception as e:
            print(f"Failed to load InstructBLIP: {e}")
        
        # Native BLIP-2
        try:
            self.inferencers['blip2'] = ModelInferencer(
                model_type="blip2"
            )
            print("Native BLIP-2 ready")
        except Exception as e:
            print(f"Failed to load BLIP-2: {e}")
        
        # LoRA model (if path provided)
        if lora_model_path and os.path.exists(lora_model_path):
            try:
                self.inferencers['lora'] = ModelInferencer(
                    model_type="lora",
                    model_path=lora_model_path
                )
                print("LoRA fine-tuned model ready")
            except Exception as e:
                print(f"Failed to load LoRA model: {e}")
        else:
            print("No LoRA model path provided, skipping")
        
        print(f"\nReady models: {list(self.inferencers.keys())}")
    
    def run_comparison(self, image_path, instruction=None):
        """Run inference comparison across all loaded models"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        print(f"\nAnalyzing image: {image_path}")
        if instruction:
            print(f"Instruction: {instruction}")
        
        results = {}
        
        for model_name, inferencer in self.inferencers.items():
            print(f"\nRunning {model_name} inference...")
            try:
                caption = inferencer.generate_caption(image_path, instruction)
                results[model_name] = caption
                print(f"{model_name}: {caption}")
            except Exception as e:
                results[model_name] = f"Error: {e}"
                print(f"{model_name}: Failed - {e}")
        
        return results
    
    def save_comparison_results(self, results, save_path):
        """Save comparison results to file"""
        import json
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {save_path}")


# Utility functions for quick inference
def quick_inference(image_path, model_type="instructblip", model_path=None, instruction=None):
    """Quick single image inference"""
    inferencer = ModelInferencer(
        model_type=model_type,
        model_path=model_path
    )
    return inferencer.generate_caption(image_path, instruction)


def compare_native_vs_lora(image_path, lora_model_path, instruction=None):
    """Quick comparison between native and LoRA models"""
    # Native model
    native_inferencer = ModelInferencer(model_type="instructblip")
    
    # LoRA model
    lora_inferencer = ModelInferencer(
        model_type="lora",
        model_path=lora_model_path
    )
    
    return native_inferencer.compare_models(
        image_path, lora_inferencer, instruction
    )