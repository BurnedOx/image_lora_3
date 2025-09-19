#!/usr/bin/env python3
"""
LoRA Inference Script for Stable Diffusion
Generate images using trained LoRA weights
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import json

def load_pipeline(model_path: str, base_model: str = "runwayml/stable-diffusion-v1-5"):
    """Load the pipeline with LoRA weights"""
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # Load LoRA weights
    adapter_model_path = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_model_path):
        # Load LoRA weights using the new method for safetensors format
        pipe.load_lora_weights(model_path)
        print(f"Loaded LoRA weights from {model_path}")
    elif os.path.exists(os.path.join(model_path, "pytorch_lora_weights.bin")):
        # Fallback to old format
        pipe.unet.load_attn_procs(model_path)
        print(f"Loaded LoRA weights from {model_path}")
    else:
        print(f"Warning: LoRA weights not found at {model_path}, using base model")
    
    # Enable optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    
    # Use DPM-Solver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    return pipe

def generate_images(
    prompt: str,
    model_path: str,
    num_images: int = 4,
    output_dir: str = "./generated",
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    negative_prompt: str = "low quality, blurry, distorted, ugly, deformed, bad anatomy",
    width: int = 512,
    height: int = 512,
):
    """Generate images with the trained LoRA"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    pipe = load_pipeline(model_path)
    
    # Set seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate images
    images = []
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}")
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        
        # Save image
        filename = f"generated_{i+1:03d}.png"
        image.save(os.path.join(output_dir, filename))
        images.append(image)
        
        # Increment seed for variety
        generator = generator.manual_seed(seed + i + 1)
    
    print(f"Generated {len(images)} images in {output_dir}")
    return images

def main():
    parser = argparse.ArgumentParser(description="Generate images with trained LoRA")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained LoRA weights")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for image generation")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="Output directory for generated images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of inference steps")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, blurry, distorted, ugly, deformed, bad anatomy",
                        help="Negative prompt")
    
    args = parser.parse_args()
    
    # Generate images
    generate_images(
        prompt=args.prompt,
        model_path=args.model_path,
        num_images=args.num_images,
        output_dir=args.output_dir,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
    )

if __name__ == "__main__":
    main()