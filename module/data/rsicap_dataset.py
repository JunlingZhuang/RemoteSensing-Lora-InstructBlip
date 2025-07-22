"""
RSICap dataset loader for LoRA fine-tuning.
Agile approach: Simple dataset class that works, can be enhanced later.
"""

import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RSICapDataset(Dataset):
    """Simple RSICap dataset loader"""
    
    def __init__(self, data, images_dir, processor, split='train'):
        self.data = data
        self.images_dir = images_dir
        self.processor = processor
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, item['filename'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # RSICap格式：使用caption作为目标文本，text_input作为指令
        instruction = item.get('text_input', 'Describe this remote sensing image.')
        target_text = item.get('caption', item.get('text_output', ''))

        # 如果没有text_input，使用默认指令
        if not instruction:
            instruction = 'Describe this remote sensing image.'
        
        # Process with InstructBLIP processor - pass both images and instruction text
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt"
        )
        
        # Process target for training
        with self.processor.tokenizer.as_target_tokenizer():
            labels = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )



        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'qformer_input_ids': inputs['qformer_input_ids'].squeeze(0),
            'qformer_attention_mask': inputs['qformer_attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0),  # Keep as sequence
            'text_input': instruction,  # 指令文本
            'text_output': target_text,  # 目标文本
            'filename': item['filename']
        }


def load_rsicap_data(config):
    """
    Load and split RSICap dataset.
    Returns train_loader, val_loader, processor
    """
    
    # Use default paths if not specified in config
    if not hasattr(config, 'rsicap_captions_file') or not config.rsicap_captions_file:
        config.rsicap_captions_file = "data/rsgpt_dataset/RSICap/captions.json"
    if not hasattr(config, 'rsicap_images_dir') or not config.rsicap_images_dir:
        config.rsicap_images_dir = "data/rsgpt_dataset/RSICap/images"
    
    print(f"Loading RSICap data from {config.rsicap_captions_file}")
    
    # Load captions file
    with open(config.rsicap_captions_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different data formats
    if 'annotations' in raw_data:
        data = raw_data['annotations']
    else:
        data = raw_data
    
    print(f"Loaded {len(data)} samples")
    
    # Import processor here to avoid circular imports
    from transformers import InstructBlipProcessor
    processor = InstructBlipProcessor.from_pretrained(config.model_name)
    
    # Split data
    train_data, val_data = train_test_split(
        data, 
        test_size=config.val_split,
        random_state=config.random_seed,
        shuffle=True
    )
    
    print(f"Split: {len(train_data)} train, {len(val_data)} validation")
    
    # Create datasets
    train_dataset = RSICapDataset(train_data, config.rsicap_images_dir, processor, 'train')
    val_dataset = RSICapDataset(val_data, config.rsicap_images_dir, processor, 'val')
    
    # Create data loaders (simple configuration that works)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, processor


class RSIEvalVQADataset(Dataset):
    """RSIEval VQA dataset loader"""
    
    def __init__(self, qa_samples, images_dir, processor):
        self.qa_samples = qa_samples
        self.images_dir = images_dir
        self.processor = processor
    
    def __len__(self):
        return len(self.qa_samples)
    
    def __getitem__(self, idx):
        sample = self.qa_samples[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, sample['filename'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process with InstructBLIP processor - use question as instruction
        inputs = self.processor(
            images=image,
            text=sample['question'],
            return_tensors="pt"
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'question': sample['question'],
            'answer': sample['answer'],
            'filename': sample['filename'],
            'type': sample['type']
        }


def load_rsieval_vqa_data(config, rsieval_path=None):
    """
    Load RSIEval VQA dataset for evaluation.
    Returns test_loader, processor, qa_samples
    """

    # Use default path if not provided
    if rsieval_path is None:
        rsieval_path = "data/rsgpt_dataset/RSIEval"

    images_dir = os.path.join(rsieval_path, "images")
    annotations_file = os.path.join(rsieval_path, "annotations.json")
    
    print(f"Loading RSIEval VQA data from {annotations_file}")
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten QA pairs
    qa_samples = []
    for annotation in data['annotations']:
        filename = annotation['filename']
        for qa_pair in annotation['qa_pairs']:
            qa_samples.append({
                'filename': filename,
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'type': qa_pair['type']
            })
    
    print(f"Loaded {len(qa_samples)} QA pairs from {len(data['annotations'])} images")
    
    # Import processor
    from transformers import InstructBlipProcessor
    processor = InstructBlipProcessor.from_pretrained(config.model_name)
    
    # Create VQA dataset
    vqa_dataset = RSIEvalVQADataset(qa_samples, images_dir, processor)
    
    # Create data loader
    test_loader = DataLoader(
        vqa_dataset,
        batch_size=1,  # Process one QA at a time for VQA
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader, processor, qa_samples


def collate_fn(batch):
    """Custom collate function for batching with padding"""
    from torch.nn.utils.rnn import pad_sequence

    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    # Pad all sequence tensors to the same length
    input_ids_list = [item['input_ids'] for item in batch]
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)

    attention_mask_list = [item['attention_mask'] for item in batch]
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    qformer_input_ids_list = [item['qformer_input_ids'] for item in batch]
    qformer_input_ids = pad_sequence(qformer_input_ids_list, batch_first=True, padding_value=0)

    qformer_attention_mask_list = [item['qformer_attention_mask'] for item in batch]
    qformer_attention_mask = pad_sequence(qformer_attention_mask_list, batch_first=True, padding_value=0)

    # Pad labels to the same length (use -100 as padding token for loss computation)
    labels_list = [item['labels'] for item in batch]
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    filenames = [item['filename'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'qformer_input_ids': qformer_input_ids,
        'qformer_attention_mask': qformer_attention_mask,
        'labels': labels,
        'text_input': [item['text_input'] for item in batch],
        'text_output': [item['text_output'] for item in batch],
        'filenames': filenames
    }
