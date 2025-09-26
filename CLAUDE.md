# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VBD (Versatile Behavior Diffusion) is a PyTorch Lightning-based machine learning framework for generalized traffic agent simulation using diffusion models. The project integrates with Waymax (Waymo's motion simulation framework) for realistic traffic scenario generation and simulation.

## Development Environment Setup

Environment setup requires conda and specific dependencies:
```bash
conda env create -n vbd -f environment.yml
conda activate vbd
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
pip install -e .
```

The project requires CUDA for GPU training but forces TensorFlow and JAX to CPU-only mode to avoid conflicts.

## Key Commands

### Data Preparation
```bash
python script/extract_data.py \
    --data_dir /path/to/waymo_open_motion_dataset_dir \
    --save_dir /path/to/data_save_dir \
    --num_workers 16 \
    --save_raw
```

### Training
```bash
python script/train.py --cfg config/VBD.yaml --num_gpus 8
```

### Testing/Simulation
```bash
python script/test.py --test_set /path/to/data --model_path ./train_log/VBD/model.pth --save_simulation
```

## Architecture Overview

### Core Components

- **VBD Model** (`vbd/model/VBD.py`): Main PyTorch Lightning module implementing the diffusion-based behavior model with three trainable components:
  - Encoder: Processes traffic scenarios 
  - Denoiser: Implements diffusion process for action generation
  - GoalPredictor: Predicts future goals/intentions

- **Data Pipeline** (`vbd/data/`): 
  - `dataset.py`: WaymaxDataset for training data loading
  - `data_utils.py`: Utility functions for data preprocessing
  - `waymax_utils.py`: Waymax-specific data handling

- **Simulation Framework** (`vbd/sim_agent/`):
  - `sim_actor.py`: VBDTest class for closed-loop simulation
  - `waymax_env.py`: Environment wrapper for Waymax integration
  - `guidance_metrics/`: Various metrics for guided generation (tracking, collision avoidance, on-road constraints)

- **Visualization** (`vbd/waymax_visualization/`): Comprehensive visualization tools for traffic scenarios and simulation results

### Configuration Management

The main configuration is in `config/VBD.yaml` which controls:
- Model architecture parameters (agents_len: 32, future_len: 80, encoder_layers: 6)
- Diffusion parameters (steps: 50, cosine scheduling)
- Training parameters (batch_size, learning rates, epochs)
- Data paths and logging settings

### Key Dependencies

- PyTorch Lightning for training framework
- Waymax for traffic simulation
- JAX/TensorFlow (CPU-only) for Waymax compatibility
- Standard ML libraries (numpy, matplotlib, mediapy)

## Project Structure

```
vbd/
├── model/           # Core diffusion model implementation
├── data/            # Data loading and preprocessing
├── sim_agent/       # Simulation and testing framework
└── waymax_visualization/  # Visualization tools

script/              # Main execution scripts
config/              # Configuration files
example/             # Jupyter notebook examples
```

## Important Notes

- The codebase forces TensorFlow and JAX to CPU-only to avoid GPU conflicts with PyTorch
- Training requires preprocessed Waymo Open Motion Dataset v1.2 in tf_example format
- Model checkpoints are saved to `./train_log/` by default
- Anchor file at `./vbd/data/cluster_64_center_dict.pkl` is required for training
- Examples are provided as Jupyter notebooks for both guided and unguided scenario generation