"""
LoRA model wrapper for InstructBLIP.
Agile approach: Simple wrapper, can be extended for more complex configurations later.
"""

import torch
from transformers import InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


class LoRAInstructBLIP:
    """Wrapper for InstructBLIP with LoRA"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load base model (following successful inference pattern)
        print(f"Loading base model: {config.model_name}")
        self.base_model = InstructBlipForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.torch_dtype == "float16" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA - 只应用到 Q-Former
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,  # Q-Former 是特征提取
            modules_to_save=None  # 不保存其他模块
        )

        # Apply LoRA only to Q-Former
        print("Applying LoRA to Q-Former only...")
        # 先尝试只对 qformer 应用 LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def forward(self, batch):
        """Custom forward implementation to avoid inputs_embeds conflict and ensure LoRA works"""
        # Move inputs to device
        pixel_values = batch['pixel_values'].to(self.device)
        qformer_input_ids = batch['qformer_input_ids'].to(self.device)
        qformer_attention_mask = batch['qformer_attention_mask'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Step 1: Vision encoding
        vision_outputs = self.model.vision_model(pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        # Check vision outputs for stability
        if torch.any(torch.isnan(image_embeds)) or torch.any(torch.isinf(image_embeds)):
            # Don't print here to avoid disrupting progress bar
            return None

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # Step 2: Q-Former processing (LoRA applied here automatically)
        # Clone query_tokens to avoid state pollution between batches
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1).clone()
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        qformer_atts = torch.cat([query_atts, qformer_attention_mask], dim=1)

        query_outputs = self.model.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Step 3: Project Q-Former outputs
        query_embeds = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]

        # Check for NaN in Q-Former outputs (critical for stability)
        if torch.any(torch.isnan(query_embeds)) or torch.any(torch.isinf(query_embeds)):
            # Don't print here to avoid disrupting progress bar
            return None  # Skip this batch

        language_model_inputs = self.model.language_projection(query_embeds)

        # Check projection outputs
        if torch.any(torch.isnan(language_model_inputs)) or torch.any(torch.isinf(language_model_inputs)):
            # Don't print here to avoid disrupting progress bar
            return None  # Skip this batch

        language_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long).to(self.device)

        # Step 4: Get text embeddings
        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        # Step 5: Concatenate embeddings
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        full_attention_mask = torch.cat([language_attention_mask, attention_mask], dim=1)

        # Adjust labels for the prepended tokens
        empty_targets = torch.ones(language_attention_mask.size(), dtype=torch.long).to(self.device).fill_(-100)
        labels = torch.cat([empty_targets, labels], dim=1)

        # Step 6: Call language model (only inputs_embeds, avoiding conflict)
        # LoRA is automatically applied in the language model layers
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs
    
    def generate(self, pixel_values, input_ids, attention_mask, **kwargs):
        """Generate text for inference (following successful inference pattern)"""
        self.model.eval()
        with torch.no_grad():
            # Use default generation settings from debug notebook
            generation_kwargs = {
                'max_new_tokens': kwargs.get('max_new_tokens', 300),
                'do_sample': kwargs.get('do_sample', True),
                'temperature': kwargs.get('temperature', 1.0),
                'top_p': kwargs.get('top_p', 0.9),
                'repetition_penalty': kwargs.get('repetition_penalty', 1.0),
                'num_beams': kwargs.get('num_beams', 1),
            }
            
            outputs = self.model.generate(
                pixel_values=pixel_values.to(self.device),
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                **generation_kwargs
            )
            
        return outputs
    
    def save_pretrained(self, save_path):
        """Save LoRA weights"""
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path):
        """Load LoRA weights"""
        # This will be implemented when needed for inference
        pass

    def verify_lora_training(self):
        """Verify that LoRA parameters are being trained correctly"""
        print("\n" + "="*60)
        print("LoRA Training Verification")
        print("="*60)

        # Check trainable parameters
        trainable_params = 0
        total_params = 0
        lora_params = {}

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    lora_params[name] = {
                        'shape': param.shape,
                        'requires_grad': param.requires_grad,
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item()
                    }

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")
        print(f"LoRA parameters found: {len(lora_params)}")

        # Show LoRA parameter details
        if lora_params:
            print("\nLoRA Parameters Details:")
            for name, info in list(lora_params.items())[:5]:  # Show first 5
                print(f"  {name}: {info['shape']}, mean={info['mean']:.6f}, std={info['std']:.6f}")
            if len(lora_params) > 5:
                print(f"  ... and {len(lora_params) - 5} more LoRA parameters")
        else:
            print("⚠️  WARNING: No LoRA parameters found!")

        # Check Q-Former specifically
        qformer_params = 0
        qformer_lora_params = 0
        for name, param in self.model.named_parameters():
            if 'qformer' in name.lower():
                qformer_params += param.numel()
                if param.requires_grad and 'lora' in name.lower():
                    qformer_lora_params += param.numel()

        print(f"\nQ-Former Analysis:")
        print(f"  Total Q-Former parameters: {qformer_params:,}")
        print(f"  Q-Former LoRA parameters: {qformer_lora_params:,}")

        if qformer_lora_params > 0:
            print("✅ Q-Former LoRA is correctly configured!")
        else:
            print("❌ Q-Former LoRA not found - check configuration!")

        print("="*60)
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def parameters(self):
        """Get model parameters for optimizer"""
        return self.model.parameters()
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "CUDA not available"