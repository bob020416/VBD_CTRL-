# VBD Test Open-loop Shape Analysis

## 1. 原始PKL數據 (load_pkl_data)
```
agents_history:     [64, 11, 8]  # 64 agents, 11 past timesteps, 8 features
agents_future:      [64, 81, 5]  # 64 agents, 81 future timesteps, 5 features  
agents_interested:  [64]         # 64 agents, boolean mask
agents_type:        [64]         # 64 agents, type classification
polylines:          [256, 30, 5] # 256 road segments, 30 points each, 5 features
traffic_lights:     [16, 3]      # 16 traffic lights, 3 features (x, y, state)
relations:          [336, 336, 3] # All entities relations (64+256+16)
```

## 2. 模型輸入準備 (prepare_model_batch)
```
agents_history:     [1, 32, 11, 8]   # Batch=1, sliced to 32 agents
agents_future:      [1, 32, 81, 5]   # Batch=1, sliced to 32 agents
agents_interested:  [1, 32]          # Batch=1, sliced to 32 agents
agents_type:        [1, 32]          # Batch=1, sliced to 32 agents  
polylines:          [1, 256, 30, 5]  # Batch=1, keep all polylines
traffic_lights:     [1, 16, 3]       # Batch=1, keep all traffic lights
relations:          [1, 304, 304, 3] # Batch=1, sliced relations (32+256+16)
anchors:            [1, 32, 64]      # Batch=1, processed anchors
```

## 3. 模型輸出
### Diffusion Mode:
```
pred['denoised_trajs']: [1, 32, 80, 3]  # Batch, 32 agents, 80 timesteps, 3 features (x,y,heading)
pred_traj:              [32, 80, 3]     # Remove batch dimension
```

### Prior Mode:  
```
pred['goal_trajs']:     [1, 32, K, 80, 3]  # Batch, 32 agents, K goals, 80 timesteps, 3 features
pred['goal_scores']:    [1, 32, K]         # Batch, 32 agents, K goal scores
sampled_idx:            [32]               # Selected goal index per agent
pred_traj:              [32, 80, 3]        # Final sampled trajectories
```

## 4. Metrics計算 (時間對齊檢查)
```
gt_future:    [64, 81, 5]     # Original GT, timesteps 0-80
pred_traj:    [32, 80, 3]     # Model prediction, timesteps 1-80

# 正確對齊:
gt_pos:       [32, 80, 2]     # gt_future[:32, 1:81, :2]  - timesteps 1-80  
pred_pos:     [32, 80, 2]     # pred_traj[:32, :80, :2]   - timesteps 1-80
```

## 5. 視覺化數據準備 (create_trajectory_video)
```
agents_history:       [64, 11, 8]   # Original history
agents_future:        [64, 81, 5]   # Original GT future
pred_future:          [64, 81, 5]   # Prepared prediction (filled with pred_traj)

# 時間masking for animation:
gt_partial[t]:        [64, 81, 5]   # gt_partial[:, t+1:] = -1
pred_partial[t]:      [64, 81, 5]   # pred_partial[:, t+1:] = -1
```

## 6. 繪圖函數 (plot_scenario_with_original)
```
agents_history:           [64, 11, 8]   # For history trajectory and vehicle dimensions
agents_future_original:   [64, 81, 5]   # For current position lookup (unmasked)
agents_future_masked:     [64, 81, 5]   # For future trajectory display (masked)
polylines:               [256, 30, 5]   # Road segments
traffic_lights:          [16, 3]        # Traffic lights

# Current timestep access:
current_pos = agents_future_original[i, current_timestep]  # Shape: [5]
```

## 7. 潛在形狀問題檢查

### ❌ 可能的問題:
1. **Relations tensor slicing**: 
   - 原始: [336, 336, 3] = (64+256+16)²
   - 目標: [304, 304, 3] = (32+256+16)²
   - 需確認索引計算正確

2. **Prediction填充**:
   - pred_traj: [32, 80, 3] 填入 pred_future: [64, 81, 5]
   - 只填入前32個agent，其他32個保持invalid (-1)
   - 時間偏移: pred_traj[t] → pred_future[t+1]

3. **Velocity計算**: 
   - 在prepare_model_batch中沒有，但在visualization中有計算
   - 可能不一致

### ✅ 正確的部分:
1. **時間對齊**: GT[1:81] vs Pred[0:80] 正確對應
2. **Agent slicing**: 前32個agent一致處理  
3. **Coordinate filtering**: 統一使用50000作為上界
4. **Invalid value handling**: 統一使用-1和0作為invalid標記