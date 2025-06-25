# ScenarioNet to VBD Conversion Script

## Overview
This script converts ScenarioNet pickle files to VBD format with automatic train/test splitting.

## Usage

### Basic Usage
```bash
python script/convert_scenarionet_to_vbd.py \
    --input_dir /path/to/scenarionet/pkl/files \
    --output_dir /path/to/vbd/output \
    --split_ratio 0.9
```

### Full Options
```bash
python script/convert_scenarionet_to_vbd.py \
    --input_dir /home/hcis-s26/Yuhsiang/HetroD/scenarionet_converter/waymo_valid/ \
    --output_dir /home/hcis-s26/Yuhsiang/VBD/data/converted_scenarionet/ \
    --split_ratio 0.9 \
    --seed 42 \
    --max_files 100
```

### Parameters
- `--input_dir`: Directory containing ScenarioNet pickle files (searches recursively)
- `--output_dir`: Output directory for VBD format files
- `--split_ratio`: Train/test split ratio (default: 0.9 = 90% train, 10% test)
- `--seed`: Random seed for reproducible splits (default: 42)
- `--max_files`: Maximum files to process for testing (optional)

## Output Structure
```
output_dir/
├── train/
│   ├── scenario_1a8c2a49578185dc.pkl
│   ├── scenario_2b9d3a5067928ef4.pkl
│   └── ...
└── test/
    ├── scenario_3c0e4b6178a3f9e5.pkl
    ├── scenario_4d1f5c7289b4ea06.pkl
    └── ...
```

## Data Format Conversion

### VBD Output Format (per pickle file)
Each converted file contains:
```python
{
    'agents_history': np.ndarray,      # (64, 11, 8) - Agent trajectories (history)
    'agents_future': np.ndarray,       # (64, 80, 5) - Agent trajectories (future)
    'agents_type': np.ndarray,         # (64,) - Agent types (1=VEHICLE, 2=PEDESTRIAN, 3=CYCLIST)
    'agents_interested': np.ndarray,   # (64,) - Interest levels (10=high, 1=low, 0=invalid)
    'polylines': np.ndarray,           # (256, 30, 5) - Map polylines [x, y, heading, traffic_state, type]
    'polylines_valid': np.ndarray,     # (256,) - Polyline validity mask
    'traffic_light_points': np.ndarray, # (16, 3) - Traffic lights [x, y, state]
    'relations': np.ndarray,           # (N, N, 3) - Spatial relations [local_x, local_y, theta_diff]
    'anchors': np.ndarray,             # (64, 64) - Agent type anchor embeddings
    'agents_id': np.ndarray,           # (64,) - Original agent IDs
    'scenario_id': str                 # Original scenario ID
}
```

### Key Transformations

1. **Temporal Splitting**:
   - History: timesteps 0-10 (11 frames)
   - Future: timesteps 11-90 (80 frames)

2. **Agent Selection**:
   - Sorts agents by distance to SDC (Self-Driving Car)
   - Takes closest 64 agents
   - Pads with zeros if fewer agents

3. **Map Processing**:
   - Filters polylines by proximity to agents
   - Resamples all polylines to 30 points
   - Takes 256 most relevant polylines

4. **Coordinate System**:
   - Preserves global coordinates
   - Computes spatial relations in local frames

## Compatibility with VBD

The output format is fully compatible with VBD's `WaymaxDataset` class:

```python
from vbd.data.dataset import WaymaxDataset

# Load converted data
dataset = WaymaxDataset(
    data_dir="/path/to/vbd/output/train",
    anchor_path="data/cluster_64_center_dict.pkl"
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

## Notes

1. **Anchor Embeddings**: The script generates dummy anchor embeddings. For production use, you should:
   - Train proper anchor clusters on your dataset
   - Replace the `generate_agent_anchors()` function
   - Save/load from `cluster_64_center_dict.pkl`

2. **Agent Interest Levels**: Currently assigns high interest to first 16 agents. You may want to:
   - Use metadata from ScenarioNet to identify important agents
   - Implement smarter interest assignment logic

3. **Memory Usage**: Processing large datasets may require significant RAM. Consider:
   - Processing in batches
   - Using `--max_files` for testing
   - Monitoring memory usage

4. **Validation**: Always validate converted data:
   ```bash
   python -c "
   import pickle
   data = pickle.load(open('output_dir/train/scenario_*.pkl', 'rb'))
   print('Data shapes:', {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in data.items()})
   "
   ```

## Example Conversion

```bash
# Convert HetroD ScenarioNet data to VBD format
python script/convert_scenarionet_to_vbd.py \
    --input_dir /home/hcis-s26/Yuhsiang/HetroD/scenarionet_converter/waymo_valid/ \
    --output_dir /home/hcis-s26/Yuhsiang/VBD/data/converted_scenarionet/ \
    --split_ratio 0.9

# Expected output:
# Found 1000 pickle files
# Split: 900 train, 100 test
# Processing train: 100%|██████████| 900/900
# train: 895 successful, 5 failed
# Processing test: 100%|██████████| 100/100  
# test: 98 successful, 2 failed
# Conversion completed!
```