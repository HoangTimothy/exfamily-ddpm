"""
Quick test script with reduced steps for debugging.
Train only 100 steps (5 FID checkpoints) instead of 20000.
"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Test config: minimal steps to debug quickly
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False  # Start with False to avoid kernel issues
)

# Option 3: Convert model to fp16, but keep buffers as fp32
model = model.half()

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000
)

# Keep diffusion buffers as float32 (for numerical stability)
# They are already float32 by default, so no change needed

trainer = Trainer(
    diffusion,
    folder="./datasets/cifar10",
    train_batch_size=64,
    train_num_steps=100,  # Very short for testing
    save_and_sample_every=20,  # Checkpoint every 20 steps
    calculate_fid=False,  # Disable FID to speed up testing
    amp=True,
    mixed_precision_type='fp16',
    num_samples=4,  # Minimal samples
    results_folder='./results_test'
)

print("Starting quick test training (100 steps)...")
trainer.train()
print("Test complete!")
