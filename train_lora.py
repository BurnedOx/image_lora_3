#!/usr/bin/env python3
"""
Image LoRA Trainer for Stable Diffusion
Optimized for 16GB VRAM
"""

import argparse
import os
import random
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig, get_peft_model
import numpy as np

from config import get_optimized_config
from dataset_utils import ImageLoRADataset, collate_fn

# Will error if the minimal version of diffusers is not installed
check_min_version("0.21.0")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training script")
    
    # Basic settings
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Path to training images")
    parser.add_argument("--output_dir", type=str, default="./lora-output")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for training (e.g., 'photo of a beautiful girl')")
    parser.add_argument("--placeholder_token", type=str, default="<girl>",
                        help="Placeholder token for the subject")
    
    # Training settings
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    
    # LoRA settings
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=27)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # System settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
    parser.add_argument("--enable_xformers", action="store_true", default=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Enable xformers
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    unet = get_peft_model(unet, lora_config)
    
    # Dataset
    train_dataset = ImageLoRADataset(
        data_root=args.train_data_dir,
        prompt=args.prompt,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available, using regular AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps
        if args.max_train_steps
        else args.num_train_epochs * len(train_dataloader) * args.gradient_accumulation_steps,
    )
    
    # Move to device
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device)
    
    # Prepare everything
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    # Training loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Compute loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Gather the losses across all processes for logging
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            
            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0
                
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        
        # Save training config
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    accelerator.end_training()

if __name__ == "__main__":
    import math
    import json
    main()