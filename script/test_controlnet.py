#!/usr/bin/env python3
"""
Simple test script to verify ControlNet implementation
"""
import sys
import os
sys.path.append('/home/hcis-s26/Yuhsiang/VBD')

import torch
import yaml
from vbd.model.vbd_ctrl import VBD as VBD_ControlNet

def test_controlnet():
    print("=== Testing ControlNet Implementation ===")

    # Load config
    config_path = "/home/hcis-s26/Yuhsiang/VBD/config/VBD.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Add ControlNet specific config
    cfg.update({
        'use_controlnet': True,
        'freeze_original': True,
        'future_feature_dim': 128,
        'film_blocks': [0, 1],
        'future_len': 81,  # Fix: (81-1) % 2 = 80 % 2 = 0 ✅
    })

    print(f"Config loaded: {list(cfg.keys())}")

    # Create model
    try:
        model = VBD_ControlNet(cfg=cfg)
        print("✅ Model created successfully")

        # Need to call configure_optimizers to trigger parameter freezing
        model.configure_optimizers()

        # Count parameters after freezing
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

        # Show which modules are trainable
        trainable_modules = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                module_name = name.split('.')[0]
                if module_name not in trainable_modules:
                    trainable_modules.append(module_name)
        print(f"Trainable modules: {trainable_modules}")

        # Test forward pass with dummy data
        batch_size = 2
        agents_len = cfg['agents_len']
        future_len = cfg['future_len']

        # Create dummy batch data
        dummy_batch = {
            'agents_future': torch.randn(batch_size, agents_len, future_len, 3),
            'agents_history': torch.randn(batch_size, agents_len, 11, 8),
            'agents_type': torch.randint(0, 4, (batch_size, agents_len)),
            'agents_interested': torch.ones(batch_size, agents_len),
            'polylines': torch.randn(batch_size, 256, 20, 5),
            'polylines_valid': torch.ones(batch_size, 256),
            'traffic_light_points': torch.randn(batch_size, 64, 3),
            'relations': torch.randn(batch_size, agents_len + 256 + 64, agents_len + 256 + 64, 3),
            'anchors': torch.randn(batch_size, agents_len, 64, 2),
        }

        print("✅ Dummy data created")

        # Test training step
        model.train()
        try:
            loss, log_dict = model.forward_and_get_loss(dummy_batch)
            print(f"✅ Training forward pass successful")
            print(f"Loss: {loss.item():.4f}")
            print(f"Metrics: {list(log_dict.keys())}")

        except Exception as e:
            print(f"❌ Training forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test evaluation step
        model.eval()
        try:
            with torch.no_grad():
                loss, log_dict = model.forward_and_get_loss(dummy_batch)
            print(f"✅ Evaluation forward pass successful")

        except Exception as e:
            print(f"❌ Evaluation forward pass failed: {e}")
            return False

        print("🎉 All tests passed! ControlNet is ready for training.")
        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_controlnet()
    sys.exit(0 if success else 1)