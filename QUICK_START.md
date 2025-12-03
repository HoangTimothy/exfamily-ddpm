# Quick Start: Diffusion Model Training & Testing

## Option 3 Implementation (Model FP16 + Buffers FP32)

### Quick Test (100 steps - ~1 minute)
```bash
python train_test_quick.py
```

This runs:
- 100 training steps (vs 20000)
- Checkpoint every 20 steps (5 total checkpoints)
- FID disabled (to avoid long sampling)
- flash_attn=False (safest for testing)
- Results in `./results_test`

### Full Training (20000 steps)
```bash
# Colab notebook:
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim=64,
    dim_mults=(1,2,4,8),
    flash_attn=False  # or True if you want to test
)

# Option 3: Convert model weights to fp16
model = model.half()
# Buffers (alphas, betas) stay as float32 (automatic)

diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000)

trainer = Trainer(
    diffusion,
    folder="./datasets/cifar10",
    train_batch_size=64,
    train_num_steps=20000,
    save_and_sample_every=1000,
    calculate_fid=True,
    num_fid_samples=5000,  # reduced from 50000
    fid_use_ddim=True,
    fid_ddim_steps=50,
    amp=True,
    mixed_precision_type='fp16'
)

trainer.train()
```

## Why Option 3 (Model FP16 + Buffers FP32)?

**Benefits:**
- ✅ Model weights in fp16 → 2x memory savings
- ✅ Model weights in fp16 → GPU throughput (faster matrix ops)
- ✅ Buffers (alphas, betas) in fp32 → numerical stability for diffusion schedules
- ✅ Sampling works correctly (fp32 buffers ensure accurate noise schedules)

**Trade-offs:**
- ⚠️ Some layers (like LayerNorm) might have precision loss in fp16
- ⚠️ Need to verify training doesn't diverge (check loss curves)

## Troubleshooting

If you still get "No available kernel" errors:
1. Set `flash_attn=False` (safest)
2. Set `calculate_fid=False` during testing
3. Use `train_test_quick.py` to debug quickly (100 steps takes ~1 min)

If training is slow:
- Enable `fid_use_ddim=True` and reduce `fid_ddim_steps` to 30-50
- Reduce `num_fid_samples` from 50000 to 5000-10000
