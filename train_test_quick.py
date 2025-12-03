"""
Quick test script with reduced steps for debugging.
Train only 100 steps (~1 minute) instead of 20000.

Uses Approach 3b: Model stays FP32, autocast forward only.
This is simpler and more stable than model.half().
"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Test config: minimal steps to debug quickly
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False  # Safest for testing
)

# Model stays fp32 (recommended approach)
# Autocast will optimize forward pass to fp16 where beneficial

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000
)

trainer = Trainer(
    diffusion,
    folder="./datasets/cifar10",
    train_batch_size=64,
    train_num_steps=100,  # Very short for testing
    save_and_sample_every=20,  # Checkpoint every 20 steps
    calculate_fid=False,  # Disable FID to speed up testing
    amp=True,  # Enable autocast for forward pass
    mixed_precision_type='fp16',
    num_samples=4,  # Minimal samples
    results_folder='./results_test'
)

print("Starting quick test training (100 steps, model fp32, autocast forward)...")
trainer.train()
print("Test complete!")
