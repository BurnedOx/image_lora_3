import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
import numpy as np
import json
from typing import List, Optional, Tuple
import random
from torchvision import transforms

class ImageLoRADataset(Dataset):
    """Dataset class for LoRA training on single subject images"""
    
    def __init__(
        self,
        data_root: str,
        prompt: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        placeholder_token: str = "<girl>",
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.placeholder_token = placeholder_token
        
        # Get all image paths
        self.data_root = Path(data_root)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.image_paths = [
            p for p in self.data_root.rglob('*') 
            if p.suffix.lower() in valid_extensions
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root}")
            
        print(f"Found {len(self.image_paths)} images")
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transforms(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.size, self.size)
        
        # Tokenize the prompt
        text = self.prompt.replace(self.placeholder_token, self.placeholder_token)
        
        return {
            "pixel_values": image,
            "text": text,
            "image_path": str(image_path)
        }

class PromptDataset(Dataset):
    """Dataset for generating validation samples"""
    
    def __init__(self, prompt: str, num_samples: int):
        self.prompt = prompt
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = idx
        return example

def collate_fn(examples):
    """Collate function for batch processing"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    texts = [example["text"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "text": texts,
    }

def prepare_dataset(
    data_dir: str,
    output_dir: str = "./processed_data",
    target_resolution: int = 512,
    max_images: Optional[int] = None
):
    """Prepare and preprocess images for training"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(data_dir, ext)))
        images.extend(glob.glob(os.path.join(data_dir, "**", ext)))
    
    if not images:
        raise ValueError(f"No images found in {data_dir}")
    
    if max_images:
        images = images[:max_images]
    
    print(f"Processing {len(images)} images...")
    
    # Process images
    processed_count = 0
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                # Convert to RGB
                img = img.convert('RGB')
                
                # Calculate resize dimensions maintaining aspect ratio
                width, height = img.size
                if width > height:
                    new_width = target_resolution
                    new_height = int(height * target_resolution / width)
                else:
                    new_height = target_resolution
                    new_width = int(width * target_resolution / height)
                
                # Resize
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Create square canvas and paste image in center
                canvas = Image.new('RGB', (target_resolution, target_resolution), (255, 255, 255))
                paste_x = (target_resolution - new_width) // 2
                paste_y = (target_resolution - new_height) // 2
                canvas.paste(img, (paste_x, paste_y))
                
                # Save processed image
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, filename)
                canvas.save(output_path, 'JPEG', quality=95)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Successfully processed {processed_count} images")
    return output_dir

def create_prompt_file(prompt: str, output_path: str = "./prompt.txt"):
    """Create a prompt file for training"""
    with open(output_path, 'w') as f:
        f.write(prompt)
    print(f"Prompt saved to {output_path}")
    return output_path