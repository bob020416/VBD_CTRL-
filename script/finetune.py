import torch
import yaml
import datetime
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# python script/finetune.py \
#        -cfg config/VBD.yaml \
#        -name VBD_ControlNet_Test \
#        -ckpt /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt \
#        -ctrl \
#        -feat_dim 128
# set tf to cpu only
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import jax
jax.config.update("jax_platform_name", "cpu")

from vbd.data.dataset import WaymaxDataset
from vbd.model.VBD import VBD as VBD_Original  # Original model
from vbd.model.vbd_ctrl import VBD as VBD_ControlNet  # ControlNet model
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from matplotlib import pyplot as plt


class MetricsComparison(pl.Callback):
    """Track and compare ControlNet vs Original model metrics"""

    def __init__(self):
        super().__init__()
        self.controlnet_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log ControlNet metrics
        current_metrics = {
            'epoch': trainer.current_epoch,
            'controlnet_loss': trainer.callback_metrics.get('val/loss', float('inf')),
            'controlnet_ade': trainer.callback_metrics.get('val/denoise_ADE', float('inf')),
            'controlnet_fde': trainer.callback_metrics.get('val/denoise_FDE', float('inf')),
        }
        self.controlnet_metrics.append(current_metrics)

        print(f"\n=== Epoch {trainer.current_epoch} ControlNet Metrics ===")
        print(f"Loss: {current_metrics['controlnet_loss']:.4f}")
        print(f"ADE: {current_metrics['controlnet_ade']:.4f}")
        print(f"FDE: {current_metrics['controlnet_fde']:.4f}")

        # Count trainable parameters
        if trainer.current_epoch == 0:
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in pl_module.parameters())
            print(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


def load_config(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def train(cfg):
    print("=== Start ControlNet Fine-tuning ===")
    print(f"ControlNet enabled: {cfg.get('use_controlnet', True)}")
    print(f"Freeze original: {cfg.get('freeze_original', True)}")
    print(f"Pretrained checkpoint: {cfg.get('ckpt_path', '/home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt')}")
    
    pl.seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")    
        
    # create dataset
    train_dataset = WaymaxDataset(
        data_dir = cfg["train_data_path"],
        anchor_path=cfg["anchor_path"],
        # max_object= cfg["agents_len"],
    )
    
    val_dataset = WaymaxDataset(
        cfg["val_data_path"],
        anchor_path=cfg["anchor_path"],
        # max_object= cfg["agents_len"],
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        pin_memory=True, 
        num_workers=cfg["num_workers"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"],
        pin_memory=True, 
        num_workers=cfg["num_workers"],
        shuffle=False
    )
    
    output_root = cfg.get("log_dir", "output")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{cfg['model_name']}_{timestamp}"
    output_path = f"{output_root}/{model_name}"
    print("Save to ", output_path)
    
    os.makedirs(output_path, exist_ok=True)
    # dump cfg to yaml file
    with open(f"{output_path}/config.yaml", "w") as file:
        yaml.dump(cfg, file)
    
    num_gpus = torch.cuda.device_count()
    print("Total GPUS:", num_gpus)

    # Configure ControlNet parameters
    controlnet_cfg = cfg.copy()
    controlnet_cfg.update({
        'use_controlnet': cfg.get('use_controlnet', True),
        'freeze_original': cfg.get('freeze_original', True),
        'future_feature_dim': cfg.get('future_feature_dim', 128),
        'film_blocks': cfg.get('film_blocks', [0, 1]),
        'future_len': cfg.get('future_len', 81),  # Ensure (future_len-1) % action_len == 0
    })

    # Create ControlNet model
    model = VBD_ControlNet(cfg=controlnet_cfg)

    # Load pretrained checkpoint (default vbd_best16.ckpt)
    ckpt_path = cfg.get("ckpt_path", "/home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt")
    if ckpt_path is not None:
        print(f"Loading pretrained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        pretrained_state = checkpoint["state_dict"]

        # Load weights into ControlNet model (excluding ControlNet-specific parts)
        model_state = model.state_dict()
        loaded_keys = []

        for key in pretrained_state:
            if key in model_state and not key.startswith('controlnet'):
                model_state[key] = pretrained_state[key]
                loaded_keys.append(key)

        model.load_state_dict(model_state)
        print(f"Loaded {len(loaded_keys)} pretrained parameters")

        # Display parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    else:
        raise ValueError("ckpt_path must be provided for ControlNet fine-tuning!")

    # Plot Scheduler
    plt.plot(model.noise_scheduler.alphas_cumprod.cpu().numpy())
    plt.plot(f"{output_path}/scheduler.jpg")
    plt.close()
    
    use_wandb = cfg.get("use_wandb", True)
    if use_wandb:
        logger = WandbLogger(
            name=model_name,
            project=cfg.get("project"),
            entity=cfg.get("username"),
            log_model=False,
            dir=output_path,
        )
    else:
        logger = CSVLogger(output_path, name="VBD", version=1, flush_logs_every_n_steps=100)
    
    gradient_clip_val = cfg.get("gradient_clip_val", 1.0)
    precision = cfg.get("precision", "bf16-mixed")
    log_every_n_steps = cfg.get("log_every_n_steps", 100)

    trainer = pl.Trainer(
        num_nodes=cfg.get("num_nodes", 1),
        max_epochs=cfg["epochs"],
        devices=cfg.get("num_gpus", -1),
        accelerator="gpu",
        strategy= DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else "auto",
        enable_progress_bar=True, 
        logger=logger, 
        enable_model_summary=True,
        detect_anomaly=False,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=20,
                save_weights_only=False,
                monitor="val/loss",
                filename="controlnet_epoch={epoch:02d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step"),
            MetricsComparison()  # Add our custom metrics tracking
        ]
    )
    print("Build Trainer")
    
    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path=cfg.get("init_from")
    )
    
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default="config/VBD.yaml")
    
    # Params for override config
    parser.add_argument("-name", "--model_name", type=str, default=None)
    parser.add_argument("-log", "-log_dir", type=str, default=None)
    
    parser.add_argument("-step", "--diffusion_steps", type=int, default=None)
    parser.add_argument("-mean", "--action_mean", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-std", "--action_std", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-zD", "--embeding_dim", type=int, default=None)
    parser.add_argument("-clamp", "--clamp_value", type=float, default=None)
    parser.add_argument("-init", "--init_from", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_ckpt", type=str, default=None)
    parser.add_argument("-nN", "--num_nodes", type=int, default=1)
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1)
    parser.add_argument("-sType", "--schedule_type", type=str, default=None)
    parser.add_argument("-sS", "--schedule_s", type=float, default=None)
    parser.add_argument("-sE", "--schedule_e", type=float, default=None)
    parser.add_argument("-scale", "--schedule_scale", type=float, default=None)
    parser.add_argument("-sT", "--schedule_tau", type=float, default=None)
    parser.add_argument("-eV", "--encoder_version", type=str, default=None)
    parser.add_argument("-pred", "--with_predictor", type=bool, default=None)
    parser.add_argument("-type", "--prediction_type", type=str, default=None)
    parser.add_argument("-gc", "--gradient_clip_val", type=float, default=None)
    parser.add_argument("-prec", "--precision", type=str, default=None)
    parser.add_argument("-les", "--log_every_n_steps", type=int, default=None)

    # ControlNet specific arguments
    parser.add_argument("-ctrl", "--use_controlnet", action="store_true", default=True,
                        help="Enable ControlNet conditioning (default: True)")
    parser.add_argument("-no_freeze", "--freeze_original", action="store_false", default=True,
                        help="Don't freeze original model parameters (default: freeze)")
    parser.add_argument("-feat_dim", "--future_feature_dim", type=int, default=128,
                        help="Future feature dimension for ControlNet (default: 128)")
    parser.add_argument("-ckpt", "--ckpt_path", type=str,
                        default="/home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt",
                        help="Path to pretrained VBD checkpoint")

    return parser
    
def load_cfg(args):
    cfg = load_config(args.cfg)
    
    # Override config from args
    # Iterate the args and override the config
    for key, value in vars(args).items():
        if key == "cfg":
            pass
        elif value is not None:
            cfg[key] = value
    return cfg
    
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    
    train(cfg)
    
