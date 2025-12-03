"""
Quick test script with reduced steps for debugging.
Train only 100 steps (5 FID checkpoints) instead of 20000.

Options:
- Use model.half() WITHOUT amp (amp=False)
  OR
- Use model as fp32 WITH amp (amp=True, autocast forward)
"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Test config: minimal steps to debug quickly
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False  # Safest for testing
)

# Option 3a: Model FP16 WITHOUT autocast (GradScaler)
# When using model.half(), must set amp=False (no GradScaler)
model = model.half()

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
    amp=False,  # IMPORTANT: disable amp when model is fp16
    num_samples=4,  # Minimal samples
    results_folder='./results_test'
)

print("Starting quick test training (100 steps, model fp16, no amp)...")
trainer.train()
print("Test complete!")
