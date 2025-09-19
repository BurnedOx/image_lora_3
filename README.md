# Image LoRA Trainer for Stable Diffusion

A complete PyTorch implementation for training LoRA (Low-Rank Adaptation) weights on Stable Diffusion models for generating realistic images of a specific subject (e.g., a girl) using personal photos.

## Features

- **Optimized for 16GB VRAM** - Efficient memory usage with gradient checkpointing and mixed precision
- **Easy setup** - Minimal configuration required
- **High quality results** - Uses best practices for LoRA training
- **Flexible prompts** - Train with custom prompts and generate variations
- **Fast inference** - Optimized generation with DPM-Solver scheduler

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (16GB VRAM recommended)
- Git

### Setup

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for 8-bit Adam optimizer:
```bash
pip install bitsandbytes
```

## Quick Start

### 1. Prepare Your Images

Place your training images (photos of the girl) in a folder:
```
data/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

### 2. Train LoRA Weights

```bash
python train_lora.py \
    --train_data_dir ./data \
    --prompt "photo of a beautiful girl" \
    --output_dir ./lora-output \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

### 3. Generate Images

```bash
python generate.py \
    --model_path ./lora-output \
    --prompt "a beautiful girl in a red dress" \
    --num_images 4 \
    --output_dir ./generated
```

## Advanced Usage

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_data_dir` | Path to training images | Required |
| `--prompt` | Training prompt | Required |
| `--placeholder_token` | Token for subject | `<girl>` |
| `--resolution` | Image resolution | 512 |
| `--train_batch_size` | Batch size | 1 |
| `--num_train_epochs` | Training epochs | 100 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--gradient_accumulation_steps` | Gradient accumulation | 4 |
| `--lora_rank` | LoRA rank | 16 |
| `--lora_alpha` | LoRA alpha | 27 |

### Optimized Configurations

For 16GB VRAM:
```bash
python train_lora.py \
    --train_data_dir ./data \
    --prompt "photo of a beautiful girl" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision fp16
```

## Tips for Best Results

### Training Tips

1. **Image Quality**: Use high-quality, diverse images of the subject
2. **Image Count**: 20-50 images work well for most subjects
3. **Consistency**: Keep the subject's face clearly visible in most images
4. **Variety**: Include different angles, lighting, and expressions
5. **Resolution**: Use images at least 512x512 pixels

### Generation Tips

1. **Prompt Engineering**: Start with simple prompts and add details gradually
2. **Guidance Scale**: 7.5-12 works well for most cases
3. **Steps**: 25-50 steps provide good quality
4. **Negative Prompt**: Always include negative prompts to avoid artifacts

## Examples

### Training Command
```bash
python train_lora.py \
    --train_data_dir ./photos \
    --prompt "photo of a beautiful girl" \
    --output_dir ./my-girl-lora \
    --num_train_epochs 100 \
    --learning_rate 1e-4
```

### Generation Commands
```bash
# Basic generation
python generate.py --model_path ./my-girl-lora --prompt "a beautiful girl smiling"

# Advanced generation
python generate.py \
    --model_path ./my-girl-lora \
    --prompt "a beautiful girl in a elegant dress, professional photography, soft lighting" \
    --num_images 8 \
    --guidance_scale 7.5 \
    --num_inference_steps 30
```

## Performance

- **Training Time**: ~2-3 hours for 100 epochs with 20-30 images on 16GB VRAM
- **Memory Usage**: ~12-14GB VRAM during training
- **Inference Time**: ~2-5 seconds per 512x512 image

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size (`--train_batch_size 1`)
- Enable gradient checkpointing (`--gradient_checkpointing`)
- Use 8-bit Adam (`--use_8bit_adam`)

**Poor Results**
- Increase training epochs
- Check image quality and variety
- Adjust learning rate (try 1e-4 to 1e-5)
- Ensure consistent prompts

## License

This project is for educational and personal use. Please respect the original Stable Diffusion license and usage terms.

## Support

If you encounter issues, please check:
1. Your GPU has sufficient VRAM (16GB recommended)
2. All dependencies are installed correctly
3. Training images are high quality and varied
4. Prompts are consistent during training