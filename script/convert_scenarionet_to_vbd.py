#!/usr/bin/env python3
"""
Convert ScenarioNet pickle files to VBD format with train/test split.

Usage:
    python convert_scenarionet_to_vbd.py --input_dir /path/to/scenarionet/pkl/files --output_dir /path/to/vbd/output --split_ratio 0.9

Output structure:
    output_dir/
    ├── train/
    │   ├── scenario_*.pkl
    │   └── ...
    └── test/
        ├── scenario_*.pkl
        └── ...
"""

import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VBD Configuration Constants
MAX_NUM_OBJECTS = 64
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30
HISTORY_LENGTH = 11  # 0 to 10
FUTURE_LENGTH = 79   # 11 to 89 (must satisfy (FUTURE_LENGTH-1) % action_len == 0)

# Agent type mapping
AGENT_TYPE_MAP = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2, 
    'CYCLIST': 3,
    'UNKNOWN': 0
}

# Traffic light state mapping
TRAFFIC_LIGHT_STATE_MAP = {
    'LANE_STATE_STOP': 0,
    'LANE_STATE_CAUTION': 1,
    'LANE_STATE_GO': 2,
    'LANE_STATE_FLASHING_STOP': 3,
    'LANE_STATE_FLASHING_CAUTION': 4,
    'LANE_STATE_UNKNOWN': 5
}

# Map feature type mapping
MAP_FEATURE_TYPE_MAP = {
    'ROAD_EDGE_BOUNDARY': 1,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 2,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 3,
    'ROAD_LINE_SOLID_DOUBLE_WHITE': 4,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW': 5,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 6,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW': 7,
    'ROAD_LINE_PASSING_DOUBLE_YELLOW': 8,
    'LANE_SURFACE_STREET': 9,
    'LANE_BIKE_LANE': 10,
    'UNKNOWN': 0
}

def wrap_to_pi(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def apply_ego_centric_transform(coordinates, ego_position, ego_heading):
    """
    Transform coordinates from global to ego-centric coordinate system.
    
    Args:
        coordinates: Array of shape (..., 2 or 3) with x, y[, z] coordinates
        ego_position: Array of shape (2,) with ego x, y position
        ego_heading: Scalar ego heading angle in radians
    
    Returns:
        Transformed coordinates in ego-centric system (same shape as input)
    """
    original_shape = coordinates.shape
    
    # Handle both 2D and 3D coordinates
    if original_shape[-1] == 2:
        # 2D coordinates (x, y)
        xy_coords = coordinates
    elif original_shape[-1] == 3:
        # 3D coordinates (x, y, z) - only transform x, y
        xy_coords = coordinates[..., :2]
    else:
        raise ValueError(f"Coordinates must have 2 or 3 dimensions, got {original_shape[-1]}")
    
    # Translate to ego position
    translated = xy_coords - ego_position
    
    # Rotate to ego heading (ego faces along positive x-axis)
    cos_theta = np.cos(-ego_heading)  # Negative for coordinate transformation
    sin_theta = np.sin(-ego_heading)
    
    # Apply rotation matrix
    translated_flat = translated.reshape(-1, 2)
    
    rotated = np.zeros_like(translated_flat)
    rotated[:, 0] = cos_theta * translated_flat[:, 0] - sin_theta * translated_flat[:, 1]
    rotated[:, 1] = sin_theta * translated_flat[:, 0] + cos_theta * translated_flat[:, 1]
    
    # Reshape back to original xy shape
    rotated = rotated.reshape(xy_coords.shape)
    
    # Handle return value based on input dimensions
    if original_shape[-1] == 2:
        return rotated
    else:
        # For 3D input, preserve z coordinate
        result = coordinates.copy()
        result[..., :2] = rotated
        return result

def transform_heading_to_ego_frame(headings, ego_heading):
    """Transform headings from global to ego-centric frame."""
    return wrap_to_pi(headings - ego_heading)

def calculate_relations(agents, polylines, traffic_lights):
    """
    Calculate the relations between agents, polylines, and traffic lights.
    """
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    # Prepare a single array to hold all elements
    all_elements = np.concatenate([
        agents[:, -1, :3],  # agents last timestep x, y, heading
        polylines[:, 0, :3],  # polylines first point x, y, heading
        np.concatenate([traffic_lights[:, :2], np.zeros((n_traffic_lights, 1))], axis=1)
    ], axis=0)

    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]

    # Compute local positions and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    # Fix: Apply correct rotation matrix for local coordinate transformation
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = pos_diff[..., 0] * (-sin_theta) + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where((np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :], 0, theta_diff)

    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Conditions for zero coordinates
    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)

    # Initialize relations array
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)

    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations

def resample_polyline(polyline_points, target_points=NUM_POINTS_POLYLINE):
    """Resample polyline to fixed number of points."""
    if len(polyline_points) == 0:
        return np.zeros((target_points, 3), dtype=np.float32)
    
    if len(polyline_points) == 1:
        # Single point, repeat it
        return np.tile(polyline_points[0], (target_points, 1))
    
    # Interpolate to target number of points
    original_indices = np.linspace(0, len(polyline_points) - 1, len(polyline_points))
    target_indices = np.linspace(0, len(polyline_points) - 1, target_points)
    
    resampled = np.zeros((target_points, 3), dtype=np.float32)
    for i in range(3):  # x, y, z coordinates
        resampled[:, i] = np.interp(target_indices, original_indices, polyline_points[:, i])
    
    return resampled

def process_agents(tracks_data):
    """Process agent data from ScenarioNet format to VBD format."""
    
    # Initialize arrays
    agents_history = np.zeros((MAX_NUM_OBJECTS, HISTORY_LENGTH, 8), dtype=np.float32)
    agents_future = np.zeros((MAX_NUM_OBJECTS, FUTURE_LENGTH, 5), dtype=np.float32)
    agents_type = np.zeros((MAX_NUM_OBJECTS,), dtype=np.int32)
    agents_interested = np.zeros((MAX_NUM_OBJECTS,), dtype=np.int32)
    agents_id = np.zeros((MAX_NUM_OBJECTS,), dtype=np.int32)
    
    # Find SDC (self-driving car) - assume first vehicle is SDC
    sdc_agent_id = None
    sdc_position = None
    sdc_heading = None
    
    for agent_id, agent_data in tracks_data.items():
        if agent_data['type'] == 'VEHICLE' and agent_data['state']['valid'][CURRENT_INDEX]:
            sdc_agent_id = agent_id
            sdc_position = agent_data['state']['position'][CURRENT_INDEX, :2]
            sdc_heading = agent_data['state']['heading'][CURRENT_INDEX]
            break
    
    if sdc_position is None:
        # Fallback: use first valid agent
        for agent_id, agent_data in tracks_data.items():
            if agent_data['state']['valid'][CURRENT_INDEX]:
                sdc_agent_id = agent_id
                sdc_position = agent_data['state']['position'][CURRENT_INDEX, :2]
                sdc_heading = agent_data['state']['heading'][CURRENT_INDEX]
                break
    
    if sdc_position is None:
        raise ValueError("No valid agent found to use as ego vehicle")
    
    # Calculate distances to SDC and sort agents by distance
    agent_distances = []
    for agent_id, agent_data in tracks_data.items():
        if agent_data['state']['valid'][CURRENT_INDEX]:
            agent_pos = agent_data['state']['position'][CURRENT_INDEX, :2]
            distance = np.linalg.norm(agent_pos - sdc_position)
            agent_distances.append((distance, agent_id))
        else:
            agent_distances.append((float('inf'), agent_id))
    
    # Sort by distance and take closest MAX_NUM_OBJECTS
    agent_distances.sort(key=lambda x: x[0])
    selected_agents = agent_distances[:MAX_NUM_OBJECTS]
    
    for i, (distance, agent_id) in enumerate(selected_agents):
        agent_data = tracks_data[agent_id]
        
        # Check if agent is valid at current timestep
        if not agent_data['state']['valid'][CURRENT_INDEX]:
            continue
            
        # Agent type
        agent_type_str = agent_data['type']
        agents_type[i] = AGENT_TYPE_MAP.get(agent_type_str, 0)
        
        # Agent interest (assume all are interested for now)
        agents_interested[i] = 10 if i < 16 else 1  # Mark first 16 as highly interested
        
        # Agent ID
        agents_id[i] = int(agent_id)
        
        # History data (timesteps 0 to CURRENT_INDEX)
        valid_mask = agent_data['state']['valid'][:HISTORY_LENGTH]
        positions = agent_data['state']['position'][:HISTORY_LENGTH]
        velocities = agent_data['state']['velocity'][:HISTORY_LENGTH]
        headings = agent_data['state']['heading'][:HISTORY_LENGTH]
        lengths = agent_data['state']['length'][:HISTORY_LENGTH]
        widths = agent_data['state']['width'][:HISTORY_LENGTH] 
        heights = agent_data['state']['height'][:HISTORY_LENGTH]
        
        # Apply ego-centric coordinate transformation
        transformed_positions = apply_ego_centric_transform(positions, sdc_position, sdc_heading)
        transformed_headings = transform_heading_to_ego_frame(headings, sdc_heading)
        
        # Transform velocities to ego frame (rotate velocity vectors - no translation)
        cos_theta = np.cos(-sdc_heading)
        sin_theta = np.sin(-sdc_heading)
        transformed_velocities = np.zeros_like(velocities)
        transformed_velocities[:, 0] = cos_theta * velocities[:, 0] - sin_theta * velocities[:, 1]
        transformed_velocities[:, 1] = sin_theta * velocities[:, 0] + cos_theta * velocities[:, 1]
        
        # Fill history: x, y, yaw, vel_x, vel_y, length, width, height
        agents_history[i, :, 0] = transformed_positions[:, 0]  # x (ego-centric)
        agents_history[i, :, 1] = transformed_positions[:, 1]  # y (ego-centric)
        agents_history[i, :, 2] = transformed_headings  # yaw (ego-centric)
        agents_history[i, :, 3] = transformed_velocities[:, 0]  # vel_x (ego-centric)
        agents_history[i, :, 4] = transformed_velocities[:, 1]  # vel_y (ego-centric)
        agents_history[i, :, 5] = lengths  # length (unchanged)
        agents_history[i, :, 6] = widths   # width (unchanged)
        agents_history[i, :, 7] = heights  # height (unchanged)
        
        # Zero out invalid timesteps
        agents_history[i][~valid_mask] = 0
        
        # Future data (timesteps CURRENT_INDEX+1 to end)
        future_start = CURRENT_INDEX + 1
        future_end = min(future_start + FUTURE_LENGTH, len(agent_data['state']['valid']))
        future_valid_mask = agent_data['state']['valid'][future_start:future_end]
        
        if future_end > future_start:
            future_positions = agent_data['state']['position'][future_start:future_end]
            future_velocities = agent_data['state']['velocity'][future_start:future_end]
            future_headings = agent_data['state']['heading'][future_start:future_end]
            
            # Apply ego-centric transformation to future data
            transformed_future_positions = apply_ego_centric_transform(future_positions, sdc_position, sdc_heading)
            transformed_future_headings = transform_heading_to_ego_frame(future_headings, sdc_heading)
            
            # Transform future velocities (rotate only, no translation)
            transformed_future_velocities = np.zeros_like(future_velocities)
            transformed_future_velocities[:, 0] = cos_theta * future_velocities[:, 0] - sin_theta * future_velocities[:, 1]
            transformed_future_velocities[:, 1] = sin_theta * future_velocities[:, 0] + cos_theta * future_velocities[:, 1]
            
            actual_future_length = future_end - future_start
            
            # Fill future: x, y, yaw, vel_x, vel_y
            agents_future[i, :actual_future_length, 0] = transformed_future_positions[:, 0]  # x (ego-centric)
            agents_future[i, :actual_future_length, 1] = transformed_future_positions[:, 1]  # y (ego-centric)
            agents_future[i, :actual_future_length, 2] = transformed_future_headings  # yaw (ego-centric)
            agents_future[i, :actual_future_length, 3] = transformed_future_velocities[:, 0]  # vel_x (ego-centric)
            agents_future[i, :actual_future_length, 4] = transformed_future_velocities[:, 1]  # vel_y (ego-centric)
            
            # Zero out invalid future timesteps
            agents_future[i, :actual_future_length][~future_valid_mask] = 0
    
    return (agents_history, agents_future, agents_interested, agents_type, agents_id, sdc_position, sdc_heading)

def process_map_features(map_features_data, agent_positions, ego_position, ego_heading):
    """Process map features from ScenarioNet format to VBD format."""
    
    # Filter and sort map features by relevance to agents
    relevant_features = []
    
    for feature_id, feature_data in map_features_data.items():
        if 'polyline' not in feature_data:
            continue
            
        polyline = feature_data['polyline']
        if len(polyline) == 0:
            continue
            
        # Calculate minimum distance to any agent
        min_distance = float('inf')
        for agent_pos in agent_positions:
            if agent_pos[0] != 0 or agent_pos[1] != 0:  # Valid agent position
                distances = np.linalg.norm(polyline[:, :2] - agent_pos[:2], axis=1)
                min_distance = min(min_distance, np.min(distances))
        
        feature_type = feature_data.get('type', 'UNKNOWN')
        type_code = MAP_FEATURE_TYPE_MAP.get(feature_type, 0)
        
        relevant_features.append((min_distance, polyline, type_code))
    
    # Sort by distance and take closest features
    relevant_features.sort(key=lambda x: x[0])
    selected_features = relevant_features[:MAX_POLYLINES]
    
    # Initialize polylines array
    polylines = np.zeros((MAX_POLYLINES, NUM_POINTS_POLYLINE, 5), dtype=np.float32)
    polylines_valid = np.zeros((MAX_POLYLINES,), dtype=np.int32)
    
    for i, (_, polyline, type_code) in enumerate(selected_features):
        # Resample polyline to fixed number of points
        resampled_polyline = resample_polyline(polyline, NUM_POINTS_POLYLINE)
        
        # Apply ego-centric coordinate transformation to polyline
        transformed_polyline = apply_ego_centric_transform(resampled_polyline[:, :2], ego_position, ego_heading)
        
        # Calculate heading from consecutive points (after transformation)
        headings = np.zeros(NUM_POINTS_POLYLINE)
        for j in range(NUM_POINTS_POLYLINE - 1):
            dx = transformed_polyline[j+1, 0] - transformed_polyline[j, 0]
            dy = transformed_polyline[j+1, 1] - transformed_polyline[j, 1]
            headings[j] = np.arctan2(dy, dx)
        headings[-1] = headings[-2]  # Copy last heading
        
        # Fill polyline: x, y, heading, traffic_light_state, type
        polylines[i, :, 0] = transformed_polyline[:, 0]  # x (ego-centric)
        polylines[i, :, 1] = transformed_polyline[:, 1]  # y (ego-centric)
        polylines[i, :, 2] = headings  # heading (ego-centric)
        polylines[i, :, 3] = 0  # traffic_light_state (will be filled later)
        polylines[i, :, 4] = type_code  # type
        
        polylines_valid[i] = 1
    
    return polylines, polylines_valid

def process_traffic_lights(dynamic_map_states_data, ego_position, ego_heading):
    """Process traffic light data from ScenarioNet format to VBD format."""
    
    traffic_light_data = []
    
    for _, light_data in dynamic_map_states_data.items():
        if light_data['type'] != 'TRAFFIC_LIGHT':
            continue
            
        stop_point = light_data['stop_point'][:2]  # x, y only
        
        # Apply ego-centric coordinate transformation to traffic light position
        transformed_stop_point = apply_ego_centric_transform(
            stop_point.reshape(1, 2), ego_position, ego_heading
        ).flatten()
        
        # Get state at current timestep
        states = light_data['state']['object_state']
        if len(states) > CURRENT_INDEX:
            state_str = states[CURRENT_INDEX]
            state_code = TRAFFIC_LIGHT_STATE_MAP.get(state_str, 5)
        else:
            state_code = 5  # Unknown
            
        traffic_light_data.append([transformed_stop_point[0], transformed_stop_point[1], state_code])
    
    # Always create array with MAX_TRAFFIC_LIGHTS size for consistent batching
    traffic_light_points = np.zeros((MAX_TRAFFIC_LIGHTS, 3), dtype=np.float32)
    
    if len(traffic_light_data) > 0:
        traffic_light_array = np.array(traffic_light_data, dtype=np.float32)
        # Take only up to MAX_TRAFFIC_LIGHTS
        num_to_copy = min(len(traffic_light_array), MAX_TRAFFIC_LIGHTS)
        traffic_light_points[:num_to_copy] = traffic_light_array[:num_to_copy]
    
    return traffic_light_points

# Note: Anchor generation removed - VBD dataset class handles this automatically

def convert_scenarionet_to_vbd(scenarionet_data):
    """Convert a single ScenarioNet pickle file to VBD format."""
    
    try:
        # Process agents with ego-centric transformation
        (agents_history, agents_future, agents_interested, 
         agents_type, agents_id, ego_position, ego_heading) = process_agents(scenarionet_data['tracks'])
        
        # Get valid agent positions for map filtering (already in ego-centric coordinates)
        valid_agents = agents_interested > 0
        agent_positions = agents_history[valid_agents, -1, :3]  # x, y, heading at current timestep
        
        # Process map features with ego-centric transformation
        polylines, polylines_valid = process_map_features(
            scenarionet_data['map_features'], agent_positions, ego_position, ego_heading)
        
        # Process traffic lights with ego-centric transformation
        traffic_light_points = process_traffic_lights(
            scenarionet_data['dynamic_map_states'], ego_position, ego_heading)
        
        # Calculate spatial relations
        relations = calculate_relations(agents_history, polylines, traffic_light_points)
        
        # Create VBD data dictionary
        # Note: anchors are not included because VBD dataset class generates them dynamically
        vbd_data = {
            'agents_history': agents_history.astype(np.float32),
            'agents_interested': agents_interested.astype(np.int32),
            'agents_type': agents_type.astype(np.int32),
            'agents_future': agents_future.astype(np.float32),
            'traffic_light_points': traffic_light_points.astype(np.float32),
            'polylines': polylines.astype(np.float32),
            'polylines_valid': polylines_valid.astype(np.int32),
            'relations': relations.astype(np.float32),
            'agents_id': agents_id.astype(np.int32)
        }
        
        # Add scenario metadata
        vbd_data['scenario_id'] = scenarionet_data['id']
        
        # Add ego transformation metadata for debugging
        vbd_data['ego_transform'] = {
            'ego_position': ego_position.astype(np.float32),
            'ego_heading': float(ego_heading),
            'transform_applied': True
        }
        
        return vbd_data
        
    except Exception as e:
        logger.error(f"Error converting scenario {scenarionet_data.get('id', 'unknown')}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert ScenarioNet pickle files to VBD format')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing ScenarioNet pickle files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for VBD format files')
    parser.add_argument('--split_ratio', type=float, default=0.9, # completely to train for ml tensor, test use simulation based testing format 
                      help='Train/test split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducible splits')
    parser.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find all pickle files
    input_path = Path(args.input_dir)
    pkl_files = list(input_path.rglob('*.pkl'))
    
    if not pkl_files:
        logger.error(f"No pickle files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(pkl_files)} pickle files")
    
    # Limit files for testing
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
        logger.info(f"Processing only {len(pkl_files)} files for testing")
    
    # Shuffle and split files
    random.shuffle(pkl_files)
    split_idx = int(len(pkl_files) * args.split_ratio)
    train_files = pkl_files[:split_idx]
    test_files = pkl_files[split_idx:]
    
    logger.info(f"Split: {len(train_files)} train, {len(test_files)} test")
    
    # Create output directories
    output_path = Path(args.output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    def process_file_list(file_list, output_dir, split_name):
        successful = 0
        failed = 0
        
        for pkl_file in tqdm(file_list, desc=f"Processing {split_name}"):
            try:
                # Load ScenarioNet data
                with open(pkl_file, 'rb') as f:
                    scenarionet_data = pickle.load(f)
                
                # Convert to VBD format
                vbd_data = convert_scenarionet_to_vbd(scenarionet_data)
                
                if vbd_data is not None:
                    # Save VBD format file
                    scenario_id = vbd_data['scenario_id']
                    output_file = output_dir / f'scenario_{scenario_id}.pkl'
                    
                    with open(output_file, 'wb') as f:
                        pickle.dump(vbd_data, f)
                    
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {pkl_file}: {str(e)}")
                failed += 1
        
        logger.info(f"{split_name}: {successful} successful, {failed} failed")
    
    # Process train and test sets
    process_file_list(train_files, train_dir, "train")
    process_file_list(test_files, test_dir, "test")
    
    logger.info("Conversion completed!")

if __name__ == '__main__':
    main()