#!/usr/bin/env python3
"""
Convert ScenarioNet format to Waymax-compatible format for testing.

This converter creates data that works with both:
1. VBD training (tensor format)
2. VBD testing (Waymax simulation)

Output format preserves all necessary information for closed-loop simulation.
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import logging

# Disable TF GPU for Waymax compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import jax
jax.config.update('jax_platform_name', 'cpu')

from waymax import datatypes
from waymax.config import ObjectType
import functools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
MAX_NUM_OBJECTS = 64
HISTORY_LENGTH = 11
FUTURE_LENGTH = 79
CURRENT_INDEX = 10

# Type mappings
SCENARIONET_TO_WAYMAX_TYPE = {
    'VEHICLE': ObjectType.SDC,  # Treat as controllable vehicle
    'PEDESTRIAN': ObjectType.PEDESTRIAN,
    'CYCLIST': ObjectType.CYCLIST,
    'UNKNOWN': ObjectType.VALID  # Generic valid object
}

SCENARIONET_TO_VBD_TYPE = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'CYCLIST': 3,
    'UNKNOWN': 0
}

def scenarionet_to_waymax_trajectory(tracks_data, scenario_length):
    """Convert ScenarioNet tracks to Waymax trajectory format."""
    
    # Sort agents by ID for consistency
    agent_ids = sorted(tracks_data.keys(), key=lambda x: int(x))
    num_agents = min(len(agent_ids), MAX_NUM_OBJECTS)
    
    # Initialize trajectory arrays
    trajectory_data = {
        'x': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'y': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'yaw': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'vel_x': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'vel_y': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'length': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'width': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'height': np.zeros((num_agents, scenario_length), dtype=np.float32),
        'valid': np.zeros((num_agents, scenario_length), dtype=bool),
        'timestamp_micros': np.arange(scenario_length, dtype=np.int64) * 100000  # 10Hz = 100ms intervals
    }
    
    # Object metadata
    object_metadata = {
        'ids': np.zeros(num_agents, dtype=np.int32),
        'object_types': np.zeros(num_agents, dtype=np.int32),
        'is_sdc': np.zeros(num_agents, dtype=bool),
        'is_modeled': np.zeros(num_agents, dtype=bool),
        'is_valid': np.zeros(num_agents, dtype=bool),
        'objects_of_interest': np.zeros(num_agents, dtype=bool)
    }
    
    # Fill trajectory data
    for i, agent_id in enumerate(agent_ids[:num_agents]):
        agent_data = tracks_data[agent_id]
        agent_state = agent_data['state']
        agent_type = agent_data['type']
        
        # Extract trajectory information
        positions = agent_state['position']  # (91, 3)
        velocities = agent_state['velocity'] # (91, 2)
        headings = agent_state['heading']    # (91,)
        valid_mask = agent_state['valid']    # (91,)
        
        # Fill arrays
        trajectory_data['x'][i] = positions[:, 0]
        trajectory_data['y'][i] = positions[:, 1]
        trajectory_data['yaw'][i] = headings
        trajectory_data['vel_x'][i] = velocities[:, 0]
        trajectory_data['vel_y'][i] = velocities[:, 1]
        trajectory_data['length'][i] = agent_state['length']
        trajectory_data['width'][i] = agent_state['width']
        trajectory_data['height'][i] = agent_state['height']
        trajectory_data['valid'][i] = valid_mask
        
        # Object metadata
        object_metadata['ids'][i] = int(agent_id)
        object_metadata['object_types'][i] = SCENARIONET_TO_WAYMAX_TYPE.get(agent_type, ObjectType.VALID)
        object_metadata['is_sdc'][i] = (i == 0)  # Treat first agent as SDC
        object_metadata['is_modeled'][i] = True   # All agents are modeled
        object_metadata['is_valid'][i] = np.any(valid_mask)
        object_metadata['objects_of_interest'][i] = True
    
    return trajectory_data, object_metadata

def scenarionet_to_waymax_roadgraph(map_features):
    """Convert ScenarioNet map features to Waymax roadgraph format."""
    
    roadgraph_points = []
    current_id = 0
    
    for feature_id, feature_data in map_features.items():
        if 'polyline' not in feature_data:
            continue
            
        polyline = feature_data['polyline']  # Shape: (N, 3)
        feature_type = feature_data.get('type', 'UNKNOWN')
        
        # Convert feature type to Waymax type code
        if 'LANE' in feature_type:
            type_code = 1  # Lane
        elif 'ROAD_LINE' in feature_type:
            type_code = 2  # Road marking
        elif 'ROAD_EDGE' in feature_type:
            type_code = 3  # Road edge
        else:
            type_code = 0  # Unknown
        
        # Add points from this polyline
        for point_idx, point in enumerate(polyline):
            # Calculate direction vector
            if point_idx < len(polyline) - 1:
                next_point = polyline[point_idx + 1]
                dir_vec = next_point[:2] - point[:2]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0:
                    dir_vec = dir_vec / dir_norm
                else:
                    dir_vec = np.array([1.0, 0.0])
            else:
                # Use previous direction for last point
                if point_idx > 0:
                    prev_point = polyline[point_idx - 1]
                    dir_vec = point[:2] - prev_point[:2]
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm > 0:
                        dir_vec = dir_vec / dir_norm
                    else:
                        dir_vec = np.array([1.0, 0.0])
                else:
                    dir_vec = np.array([1.0, 0.0])
            
            roadgraph_points.append({
                'x': point[0],
                'y': point[1], 
                'z': point[2] if len(point) > 2 else 0.0,
                'dir_x': dir_vec[0],
                'dir_y': dir_vec[1],
                'dir_z': 0.0,
                'types': type_code,
                'ids': current_id,
                'valid': True
            })
        
        current_id += 1
    
    # Convert to arrays
    if roadgraph_points:
        roadgraph_data = {
            'x': np.array([p['x'] for p in roadgraph_points], dtype=np.float32),
            'y': np.array([p['y'] for p in roadgraph_points], dtype=np.float32),
            'z': np.array([p['z'] for p in roadgraph_points], dtype=np.float32),
            'dir_x': np.array([p['dir_x'] for p in roadgraph_points], dtype=np.float32),
            'dir_y': np.array([p['dir_y'] for p in roadgraph_points], dtype=np.float32),
            'dir_z': np.array([p['dir_z'] for p in roadgraph_points], dtype=np.float32),
            'types': np.array([p['types'] for p in roadgraph_points], dtype=np.int32),
            'ids': np.array([p['ids'] for p in roadgraph_points], dtype=np.int32),
            'valid': np.array([p['valid'] for p in roadgraph_points], dtype=bool)
        }
    else:
        # Empty roadgraph
        roadgraph_data = {
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'dir_x': np.array([], dtype=np.float32),
            'dir_y': np.array([], dtype=np.float32),
            'dir_z': np.array([], dtype=np.float32),
            'types': np.array([], dtype=np.int32),
            'ids': np.array([], dtype=np.int32),
            'valid': np.array([], dtype=bool)
        }
    
    return roadgraph_data

def scenarionet_to_waymax_traffic_lights(dynamic_map_states, scenario_length):
    """Convert ScenarioNet traffic lights to Waymax format."""
    
    traffic_lights = []
    
    for light_id, light_data in dynamic_map_states.items():
        if light_data['type'] != 'TRAFFIC_LIGHT':
            continue
            
        stop_point = light_data['stop_point']
        states = light_data['state']['object_state']
        
        # Convert state strings to integers
        state_sequence = []
        for state_str in states:
            if 'GO' in state_str:
                state_code = 1
            elif 'CAUTION' in state_str:
                state_code = 2
            elif 'STOP' in state_str:
                state_code = 0
            else:
                state_code = 0  # Default to stop
            state_sequence.append(state_code)
        
        # Pad sequence to scenario length
        while len(state_sequence) < scenario_length:
            state_sequence.append(state_sequence[-1] if state_sequence else 0)
        
        traffic_lights.append({
            'lane_id': int(light_id),
            'x': stop_point[0],
            'y': stop_point[1],
            'z': stop_point[2] if len(stop_point) > 2 else 0.0,
            'state_sequence': state_sequence[:scenario_length]
        })
    
    if traffic_lights:
        # Convert to Waymax format
        num_lights = len(traffic_lights)
        traffic_light_data = {
            'lane_ids': np.array([[tl['lane_id']] * scenario_length for tl in traffic_lights], dtype=np.int32),
            'state': np.array([tl['state_sequence'] for tl in traffic_lights], dtype=np.int32),
            'x': np.array([[tl['x']] * scenario_length for tl in traffic_lights], dtype=np.float32),
            'y': np.array([[tl['y']] * scenario_length for tl in traffic_lights], dtype=np.float32),
            'z': np.array([[tl['z']] * scenario_length for tl in traffic_lights], dtype=np.float32),
            'valid': np.ones((num_lights, scenario_length), dtype=bool)
        }
    else:
        # Empty traffic lights
        traffic_light_data = {
            'lane_ids': np.array([[]], dtype=np.int32).reshape(0, scenario_length),
            'state': np.array([[]], dtype=np.int32).reshape(0, scenario_length), 
            'x': np.array([[]], dtype=np.float32).reshape(0, scenario_length),
            'y': np.array([[]], dtype=np.float32).reshape(0, scenario_length),
            'z': np.array([[]], dtype=np.float32).reshape(0, scenario_length),
            'valid': np.array([[]], dtype=bool).reshape(0, scenario_length)
        }
    
    return traffic_light_data

def create_waymax_compatible_scenario(scenarionet_data):
    """Convert ScenarioNet data to Waymax-compatible format."""
    
    scenario_length = scenarionet_data['length']
    scenario_id = scenarionet_data['id']
    
    # Convert trajectory data
    trajectory_data, object_metadata = scenarionet_to_waymax_trajectory(
        scenarionet_data['tracks'], scenario_length)
    
    # Convert roadgraph data
    roadgraph_data = scenarionet_to_waymax_roadgraph(scenarionet_data['map_features'])
    
    # Convert traffic lights
    traffic_light_data = scenarionet_to_waymax_traffic_lights(
        scenarionet_data['dynamic_map_states'], scenario_length)
    
    # Create the combined data structure
    waymax_compatible_data = {
        # Original ScenarioNet data for reference
        'scenario_id': scenario_id,
        'scenario_length': scenario_length,
        
        # Waymax-compatible trajectory data
        'log_trajectory': trajectory_data,
        'object_metadata': object_metadata,
        'roadgraph_points': roadgraph_data,
        'log_traffic_light': traffic_light_data,
        
        # Additional metadata for VBD compatibility
        'current_time_index': CURRENT_INDEX,
        'max_num_objects': MAX_NUM_OBJECTS,
        
        # Pre-computed VBD tensors for efficiency (optional)
        'vbd_precomputed': None  # Can be filled later if needed
    }
    
    return waymax_compatible_data

def convert_scenarionet_folder(input_dir, output_dir, split_ratio=0.9, max_files=None):
    """Convert a folder of ScenarioNet files to Waymax-compatible format."""
    
    # Find all pickle files
    input_path = Path(input_dir)
    pkl_files = list(input_path.rglob('*.pkl'))
    
    if not pkl_files:
        logger.error(f"No pickle files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pkl_files)} pickle files")
    
    # Limit files for testing
    if max_files:
        pkl_files = pkl_files[:max_files]
        logger.info(f"Processing only {len(pkl_files)} files for testing")
    
    # Shuffle and split files
    random.shuffle(pkl_files)
    split_idx = int(len(pkl_files) * split_ratio)
    train_files = pkl_files[:split_idx]
    test_files = pkl_files[split_idx:]
    
    logger.info(f"Split: {len(train_files)} train, {len(test_files)} test")
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file_list(file_list, output_dir, split_name):
        successful = 0
        failed = 0
        
        for pkl_file in tqdm(file_list, desc=f"Converting {split_name}"):
            try:
                # Load ScenarioNet data
                with open(pkl_file, 'rb') as f:
                    scenarionet_data = pickle.load(f)
                
                # Convert to Waymax-compatible format
                waymax_data = create_waymax_compatible_scenario(scenarionet_data)
                
                # Save converted file
                scenario_id = waymax_data['scenario_id']
                output_file = output_dir / f'waymax_{scenario_id}.pkl'
                
                with open(output_file, 'wb') as f:
                    pickle.dump(waymax_data, f)
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to convert {pkl_file}: {str(e)}")
                failed += 1
        
        logger.info(f"{split_name}: {successful} successful, {failed} failed")
    
    # Process train and test sets
    process_file_list(train_files, train_dir, "train")
    process_file_list(test_files, test_dir, "test")
    
    logger.info("Conversion completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert ScenarioNet to Waymax-compatible format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing ScenarioNet pickle files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for Waymax-compatible files')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                       help='Train/test split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Converting ScenarioNet data from {args.input_dir}")
    logger.info(f"Output will be saved to {args.output_dir}")
    
    convert_scenarionet_folder(
        args.input_dir,
        args.output_dir, 
        args.split_ratio,
        args.max_files
    )

if __name__ == '__main__':
    main()