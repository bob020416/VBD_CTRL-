# ControlNet-Style VBD Training

This guide explains how to use the modified `finetune.py` script to train the ControlNet-enhanced VBD model with future trajectory context injection.

## Quick Start

```bash
# Basic ControlNet training with 20% future context
python script/finetune.py \
    --cfg config/VBD.yaml \
    --model_name VBD_ControlNet_20pct \
    --future_subsample_ratio 0.2
```

## Key Features

### ✅ **ControlNet Architecture**
- Freezes all original VBD parameters (encoder, denoiser, predictor)
- Only trains the new `FutureTrajectoryEncoder` module
- Zero-initialized output projection for stable training

### ✅ **Future Context Injection**
- Injects 20% subsampled future trajectory information into Key/Value attention
- Configurable subsample ratio (15%, 20%, 25%, etc.)
- Maintains compatibility with existing loss functions

### ✅ **Performance Tracking**
- Automatic comparison with original model performance
- Logs improvements in loss, ADE, and FDE metrics
- CSV/WandB logging support

## Command Line Arguments

### Required Arguments
```bash
--cfg PATH                      # Path to VBD config file
```

### Optional Arguments
```bash
--pretrained_ckpt PATH          # Path to pretrained VBD checkpoint (default: /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt)
```

### ControlNet-Specific Arguments
```bash
--future_subsample_ratio 0.2    # Future context ratio (default: 20%)
--use_future_context            # Enable future context (default: True)
--freeze_original               # Freeze original parameters (default: True)
--enable_comparison             # Enable performance comparison (default: False)
```

### Training Arguments
```bash
--model_name NAME               # Output model name
--log_dir PATH                  # Output directory
--epochs N                      # Number of training epochs
--num_gpus N                    # Number of GPUs to use
```

## Example Training Commands

### 1. Basic ControlNet Training (20% context)
```bash
python script/finetune.py \
    --cfg config/VBD.yaml \
    --model_name VBD_ControlNet_20pct \
    --future_subsample_ratio 0.2 \
    --epochs 50
```

### 2. Experiment with Different Context Ratios
```bash
# 15% future context
python script/finetune.py \
    --cfg config/VBD.yaml \
    --model_name VBD_ControlNet_15pct \
    --future_subsample_ratio 0.15

# 25% future context
python script/finetune.py \
    --cfg config/VBD.yaml \
    --model_name VBD_ControlNet_25pct \
    --future_subsample_ratio 0.25
```

### 3. Training with Performance Comparison
```bash
python script/finetune.py \
    --cfg config/VBD.yaml \
    --model_name VBD_ControlNet_with_comparison \
    --future_subsample_ratio 0.2 \
    --enable_comparison \
    --epochs 100
```

### 4. Using Custom Pretrained Checkpoint
```bash
python script/finetune.py \
    --cfg config/VBD.yaml \
    --pretrained_ckpt /path/to/your/checkpoint.ckpt \
    --model_name VBD_ControlNet_custom \
    --future_subsample_ratio 0.2
```

## Expected Output

### Training Logs
```
=== Start ControlNet Fine-tuning ===
Future context enabled: True
Subsample ratio: 0.2
Pretrained checkpoint: /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt

Loading pretrained weights from: /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt
Loaded 1,234,567 pretrained parameters
Trainable parameters: 12,345 / 1,246,912 (0.99%)

=== Epoch 5 Performance Comparison ===
ControlNet - Loss: 2.1543, ADE: 1.2345, FDE: 2.3456
```

### Performance Expectations

**Theoretical improvements with 20% future context:**
- **Loss**: 5-15% reduction
- **ADE**: 10-20% improvement
- **FDE**: 15-25% improvement

**Training efficiency:**
- Only ~1% of parameters trainable
- Faster convergence (10-20 epochs)
- Lower memory usage

## Output Files

After training, you'll find:
```
output/VBD_ControlNet_20241217123456/
├── config.yaml                           # Training configuration
├── logs/                                  # CSV logs
├── controlnet_epoch=01.ckpt              # Checkpoints
├── controlnet_epoch=02.ckpt
└── scheduler.jpg                          # Noise scheduler plot
```

## Usage After Training

### Load Trained ControlNet Model
```python
from vbd.model.vbd_ctrl import VBD

# Load config
cfg = yaml.safe_load(open('config/VBD.yaml'))
cfg.update({
    'use_future_context': True,
    'future_subsample_ratio': 0.2,
    'freeze_original': False  # Set to False for inference
})

# Load trained model
model = VBD(cfg)
checkpoint = torch.load('output/VBD_ControlNet_20241217123456/controlnet_epoch=50.ckpt')
model.load_state_dict(checkpoint['state_dict'])
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   --batch_size 16
   ```

2. **Checkpoint loading errors**
   ```bash
   # Make sure pretrained checkpoint path is correct
   ls -la /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt
   ```

3. **No improvement in performance**
   - Try different subsample ratios (0.15, 0.25, 0.3)
   - Increase training epochs
   - Check data quality and validity

### Validation Commands

```bash
# Check if ControlNet model loads correctly
python -c "
from vbd.model.vbd_ctrl import VBD
cfg = {'use_future_context': True, 'future_subsample_ratio': 0.2, 'freeze_original': True}
model = VBD(cfg)
print('✓ ControlNet model created successfully')
"
```

## Performance Analysis

### Expected Training Progress
- **Epoch 1-5**: Initial adaptation to future context
- **Epoch 5-15**: Rapid improvement in metrics
- **Epoch 15+**: Fine-tuning and convergence

### Key Metrics to Track
- `val/loss`: Should decrease faster than original training
- `val/denoise_ADE`: Average displacement error
- `val/denoise_FDE`: Final displacement error
- `comparison/*_improvement_pct`: Percentage improvements

## Next Steps

1. **Ablation Studies**: Test different subsample ratios
2. **Architecture Experiments**: Try different future encoder architectures
3. **Evaluation**: Run comprehensive evaluation on test sets
4. **Deployment**: Use best model for inference and simulation