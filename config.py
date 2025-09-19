import torch
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LoRAConfig:
    """Configuration for LoRA training parameters"""
    
    # Model settings
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None
    
    # Dataset settings
    train_data_dir: str = "./data"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    
    # Training settings
    output_dir: str = "./lora-output"
    seed: int = 42
    train_batch_size: int = 1
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    dataloader_num_workers: int = 0
    
    # LoRA settings
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 27
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_text_encoder_r: int = 16
    lora_text_encoder_alpha: int = 17
    lora_text_encoder_dropout: float = 0.1
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_dir: str = "./logs"
    mixed_precision: str = "fp16"
    report_to: str = "tensorboard"
    save_steps: int = 500
    save_total_limit: int = 1
    checkpointing_steps: int = 500
    validation_steps: int = 100
    
    # Validation
    validation_prompt: Optional[str] = "a photo of a beautiful girl"
    num_validation_images: int = 4
    validation_epochs: int = 50
    
    # System settings
    allow_tf32: bool = True
    use_8bit_adam: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    set_grads_to_none: bool = True
    
    # Instance prompt settings
    instance_prompt: str = "photo of a beautiful girl"
    class_prompt: Optional[str] = None
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    
    # Metadata
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

# Optimized config for 16GB VRAM
def get_optimized_config():
    config = LoRAConfig()
    config.train_batch_size = 1
    config.gradient_accumulation_steps = 8
    config.gradient_checkpointing = True
    config.mixed_precision = "fp16"
    config.enable_xformers_memory_efficient_attention = True
    config.use_8bit_adam = True
    return config