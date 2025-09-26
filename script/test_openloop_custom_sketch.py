# VBD Open-loop Testing with Video Visualization
# Usage example:
#   python script/test_openloop_custom_sketch.py \
#       --test_path ./data/processed/Test \
#       --model_path /home/hcis-s26/Yuhsiang/VBD/ckpt/vbd_best16.ckpt \
#       --ctrl_ckpt_path /home/hcis-s26/Yuhsiang/VBD/train_log/VBD_ControlNet_Test_20250924040157/controlnet_epoch=15.ckpt \
#       --use_training_format --max_scenarios 30 --save_simulation

import argparse
import csv
import glob
import os
import pickle
from typing import Dict, List, Tuple

import jax
import matplotlib.transforms as transforms
import mediapy
import numpy as np
import tensorflow as tf
import torch
from matplotlib import pyplot as plt

from vbd.data.dataset import WaymaxDataset
from vbd.model.utils import set_seed
from vbd.sim_agent.sim_actor import VBDTest

# Force CPU execution for TF/JAX dependencies (mirrors training behaviour)
tf.config.set_visible_devices([], 'GPU')
jax.config.update('jax_platform_name', 'cpu')

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
N_SIM_AGENTS = 32
VIEW_RANGE = 75           # Camera view range (meters)
COORD_LIMIT = 100_000      # Filter extreme coordinates when computing metrics

# Helper dataset used to generate anchors identical to the training pipeline
train_dataset = WaymaxDataset(
    data_dir=None,
    anchor_path='./vbd/data/cluster_64_center_dict.pkl',
)

from vbd.model.vbd_ctrl import VBD as ControlNetModule
from vbd.sim_agent.utils import duplicate_batch, torch_dict_to_numpy, stack_dict
from tqdm import tqdm

# This class is custom for inference with ControlNet VBD we trained in finetune.py
class VBDCtrlTest(ControlNetModule):
    def __init__(
        self,
        cfg: dict,
        early_stop: int = 0,
        skip: int = 1,
        reward_func=None,
        guidance_iter: int = 5,
        guidance_end: int = 1,
        guidance_start: int = 99,
        gradient_scale: float = 1.0,
        scale_grad_by_std: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.early_stop = early_stop
        self.skip = skip
        self.reward_func = reward_func
        self.guidance_iter = guidance_iter
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.gradient_scale = gradient_scale
        self.scale_grad_by_std = scale_grad_by_std
        self.guidance_func = None

    def reset_agent_length(self, _agents_len: int) -> None:
        self._agents_len = _agents_len
        if self.predictor is not None:
            self.predictor.reset_agent_length(_agents_len)
        if self.denoiser is not None:
            self.denoiser.reset_agent_length(_agents_len)

    def inference_predictor(self, batch) -> dict:
        if self.predictor is None:
            raise RuntimeError("Predictor is not defined")
        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)
        return self.forward_predictor(encoder_outputs)

    def step_denoiser(
        self,
        x_t: torch.Tensor,
        c: dict,
        t: int,
        control_agents_future: torch.Tensor | None = None,
        control_mask: torch.Tensor | None = None,
    ):
        """
        Single diffusion step to obtain x_{t-1} given x_t and encoder outputs.
        """
        if self.denoiser is None:
            raise RuntimeError("Denoiser is not defined")

        # Conditioning only affects agents with valid mask entries; others remain baseline.
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
            agents_future=(
                control_agents_future[:, :self._agents_len]
                if control_agents_future is not None
                else None
            ),
            control_mask=(
                control_mask[:, :self._agents_len]
                if control_mask is not None
                else None
            ),
        )

        x_0 = denoiser_output['denoised_actions_normalized']
        x_t_prev = self.noise_scheduler.step(
            model_output=x_0,
            timesteps=t,
            sample=x_t,
            prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
        )

        return denoiser_output, x_t_prev

    @torch.no_grad()
    def sample_denoiser(
        self,
        batch,
        num_samples: int = 1,
        x_t = None,
        use_tqdm: bool = True,
        fix_t: int = -1,
        calc_loss: bool = False,
        **kwargs,
    ) -> dict:
        batch = self.batch_to_device(batch, self.device)
        conditioning_future = None
        control_mask = None
        if 'agents_future' in batch:
            conditioning_future = batch['agents_future'][:, :self._agents_len]
            # Build mask so only agents with a real sketch get modulated.
            valid_steps = conditioning_future[..., :2].ne(-1).any(dim=-1)
            control_mask = valid_steps.any(dim=-1).float()
        if calc_loss:
            agents_future = batch['agents_future'][:, :self._agents_len]
            agents_future_valid = torch.ne(agents_future.sum(-1), 0)
            agents_interested = batch['agents_interested'][:, :self._agents_len]
        encoder_outputs = self.encoder(batch)
        if num_samples > 1:
            encoder_outputs = duplicate_batch(encoder_outputs, num_samples)
            if conditioning_future is not None:
                conditioning_future = conditioning_future.repeat(num_samples, 1, 1, 1)
            if control_mask is not None:
                control_mask = control_mask.repeat(num_samples, 1)
        agents_history = encoder_outputs['agents']
        num_batch, num_agent = agents_history.shape[:2]
        num_step = self._future_len // self._action_len
        action_dim = 2
        diffusion_steps = list(reversed(range(self.early_stop, self.noise_scheduler.num_steps, self.skip)))
        x_t_history = []
        denoiser_output_history = []
        guide_history = []
        if x_t is None:
            x_t = torch.randn(num_batch, num_agent, num_step, action_dim, device=self.device)
        else:
            x_t = x_t.to(self.device)
        iterator = tqdm(diffusion_steps, total=len(diffusion_steps), desc="Diffusion") if use_tqdm else diffusion_steps
        for t in iterator:
            x_t_history.append(x_t.detach().cpu().numpy())
            denoiser_outputs, x_t = self.step_denoiser(
                x_t=x_t,
                c=encoder_outputs,
                t=fix_t if fix_t >= 0 else t,
                control_agents_future=conditioning_future,
                control_mask=control_mask,
            )
            if calc_loss:
                denoised_trajs = denoiser_outputs['denoised_trajs']
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )
                denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                    denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
                )
                denoiser_outputs.update(
                    {
                        'state_loss_mean': state_loss_mean,
                        'yaw_loss_mean': yaw_loss_mean,
                        'denoise_ade': denoise_ade,
                        'denoise_fde': denoise_fde,
                    }
                )
            denoiser_output_history.append(torch_dict_to_numpy(denoiser_outputs))
        denoiser_outputs['history'] = {
            'x_t_history': np.stack(x_t_history, axis=0),
            'denoiser_output_history': stack_dict(denoiser_output_history),
            'guide_history': stack_dict(guide_history),
        }
        return denoiser_outputs


# -----------------------------------------------------------------------------
# PKL utilities
# -----------------------------------------------------------------------------
def load_pkl_files(data_dir: str) -> List[str]:
    """Return a sorted list of PKL files inside *data_dir*."""
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    print(f"Found {len(pkl_files)} pkl files in {data_dir}")
    return pkl_files


def load_pkl_data(pkl_file: str) -> dict:
    """Load a preprocessed scenario stored as a PKL file."""
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# Batch preparation
# -----------------------------------------------------------------------------
def prepare_model_batch(pkl_data: dict) -> dict:
    """Convert processed PKL data into a batch compatible with the model."""
    batch: dict[str, torch.Tensor] = {}

    agent_hist = pkl_data.get('agents_history')
    orig_agent_count = agent_hist.shape[0] if agent_hist is not None else 0
    n_agents = min(N_SIM_AGENTS, orig_agent_count if orig_agent_count else N_SIM_AGENTS)
    n_polylines = pkl_data.get('polylines', np.zeros((0,))).shape[0] if 'polylines' in pkl_data else 0
    n_traffic_lights = (
        pkl_data.get('traffic_light_points', np.zeros((0,))).shape[0]
        if 'traffic_light_points' in pkl_data
        else 0
    )

    keys = [
        'agents_history', 'agents_interested', 'agents_type', 'agents_future',
        'traffic_light_points', 'polylines', 'polylines_valid', 'relations'
    ]

    for key in keys:
        if key not in pkl_data:
            continue

        data = pkl_data[key]

        if key in {'agents_history', 'agents_interested', 'agents_type', 'agents_future'}:
            data = data[:n_agents]

        elif key == 'relations':
            total = data.shape[0]
            if total != data.shape[1]:
                raise ValueError(f"relations tensor must be square, got {data.shape}")

            agent_indices = list(range(min(n_agents, total)))
            poly_start = orig_agent_count
            poly_end = min(poly_start + n_polylines, total)
            traffic_start = poly_end
            traffic_end = min(traffic_start + n_traffic_lights, total)

            keep_indices = agent_indices
            keep_indices += list(range(poly_start, poly_end))
            keep_indices += list(range(traffic_start, traffic_end))

            if not keep_indices:
                raise ValueError('Unable to derive indices for relations tensor slicing')

            data = data[np.ix_(keep_indices, keep_indices)]

        tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        batch[key] = tensor

    if 'agents_type' in pkl_data:
        anchors_source = pkl_data['agents_type'][:n_agents]
        anchors = train_dataset._process(anchors_source)
        batch['anchors'] = torch.from_numpy(np.ascontiguousarray(anchors)).unsqueeze(0)

    return batch


def override_future_with_forward_step(
    batch: dict[str, torch.Tensor],
    distance: float = 5.0,
    target_agents: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Overwrite selected agents with a forward sketch while leaving others untouched."""

    if 'agents_future' not in batch or 'agents_history' not in batch:
        return

    agents_future = batch['agents_future']  # [B, A, T, C]
    agents_history = batch['agents_history']  # [B, A, H, D]

    if agents_future.numel() == 0 or agents_history.numel() == 0:
        return

    batch_size, num_agents, future_len, future_dim = agents_future.shape

    # Filter the requested agent indices so we only touch valid entries.
    valid_indices = [idx for idx in target_agents if 0 <= idx < num_agents]
    if not valid_indices:
        return

    device = agents_future.device
    # Track which agents receive the synthetic sketch; used later to blank out others.
    selected_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
    selected_mask[:, valid_indices] = True

    current = agents_history[:, :, -1, :]  # latest history state per agent [B, A, D]
    # Offsets along the heading direction, spaced uniformly from 0 to `distance` metres.
    steps = torch.linspace(0.0, distance, future_len, device=device).view(1, 1, future_len)

    selected_current = current[:, valid_indices]  # state of the agents we override [B, |idx|, D]
    heading = selected_current[..., 2]
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)

    start_x = selected_current[..., 0].unsqueeze(-1)
    start_y = selected_current[..., 1].unsqueeze(-1)

    traj_x = start_x + cos_h * steps
    traj_y = start_y + sin_h * steps
    traj_yaw = heading.unsqueeze(-1).expand(-1, -1, future_len)

    # Inject the custom sketch (x, y, yaw) for each target agent.
    agents_future[:, valid_indices, :, 0] = traj_x
    agents_future[:, valid_indices, :, 1] = traj_y
    agents_future[:, valid_indices, :, 2] = traj_yaw

    # Clear any trailing channels (velocity, size, etc.) so no stale values leak downstream.
    if future_dim > 3:
        agents_future[:, valid_indices, :, 3:] = 0.0

    if 'agents_interested' in batch:
        uninterested_mask = batch['agents_interested'] <= 0
    else:
        uninterested_mask = current[..., :2].abs().sum(-1) <= 0

    if uninterested_mask.any():
        fill = torch.full_like(agents_future, -1.0) # fill with -1, so after the FilM layer all future becomes 0 
        # Only blank out uninterested agents that we did not explicitly target.
        mask = uninterested_mask & (~selected_mask)
        if mask.any():
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            agents_future[:] = torch.where(mask, fill, agents_future)

    # Finally, force all non-target agents to -1 so ControlNet/visualization only see the selected ones.
    non_target_mask = ~selected_mask
    if non_target_mask.any():
        fill = torch.full_like(agents_future, -1.0)
        mask = non_target_mask.unsqueeze(-1).unsqueeze(-1)
        agents_future[:] = torch.where(mask, fill, agents_future)

# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------
def plot_scenario_with_original(
    agents_history: np.ndarray,
    agents_future_original: np.ndarray,
    agents_future_masked: np.ndarray,
    polylines: np.ndarray,
    traffic_lights: np.ndarray,
    current_timestep: int = 0,
    debug: bool = False,
) -> None:
    """Plot ground-truth or predicted trajectories for a single frame."""

    for i in range(polylines.shape[0]):
        if polylines[i, 0, 0] != 0:
            line = polylines[i]
            valid_mask = (
                (line[:, 0] != 0)
                & (np.abs(line[:, 0]) < COORD_LIMIT)
                & (np.abs(line[:, 1]) < COORD_LIMIT)
            )
            if np.any(valid_mask):
                plt.plot(line[valid_mask][:, 0], line[valid_mask][:, 1], '.', markersize=1, color='gray', alpha=0.4)

    drawn_boxes = 0
    for i in range(agents_history.shape[0]):
        if agents_history[i, -1, 0] == 0:
            continue

        history = agents_history[i]
        valid_mask = (
            (history[:, 0] != 0)
            & (np.abs(history[:, 0]) < COORD_LIMIT)
            & (np.abs(history[:, 1]) < COORD_LIMIT)
        )
        if np.any(valid_mask):
            plt.plot(history[valid_mask][:, 0], history[valid_mask][:, 1], 'b-', alpha=0.5, linewidth=1)

        if current_timestep >= agents_future_original.shape[1]:
            continue

        current_pos = agents_future_original[i, current_timestep]
        if (
            current_pos[0] == 0 or current_pos[0] == -1
            or np.abs(current_pos[0]) >= COORD_LIMIT
            or np.abs(current_pos[1]) >= COORD_LIMIT
        ):
            continue

        pos_x, pos_y = current_pos[0], current_pos[1]

        if len(current_pos) > 2 and current_pos[2] != 0:
            heading = current_pos[2]
        elif current_timestep > 0:
            prev_pos = agents_future_original[i, current_timestep - 1]
            if prev_pos[0] != 0 and prev_pos[0] != -1:
                dx, dy = pos_x - prev_pos[0], pos_y - prev_pos[1]
                heading = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else agents_history[i, -1, 2]
            else:
                heading = agents_history[i, -1, 2]
        else:
            heading = agents_history[i, -1, 2]

        length = agents_history[i, -1, 5]
        width = agents_history[i, -1, 6]
        rect = plt.Rectangle(
            (pos_x - length / 2, pos_y - width / 2),
            length,
            width,
            linewidth=2,
            color='red' if i == 0 else 'orange',
            alpha=0.8,
            zorder=4,
            transform=transforms.Affine2D().rotate_around(pos_x, pos_y, heading) + plt.gca().transData,
        )
        plt.gca().add_patch(rect)
        drawn_boxes += 1

    for i in range(traffic_lights.shape[0]):
        if (
            traffic_lights[i, 0] != 0
            and np.abs(traffic_lights[i, 0]) < COORD_LIMIT
            and np.abs(traffic_lights[i, 1]) < COORD_LIMIT
        ):
            circle = plt.Circle((traffic_lights[i, 0], traffic_lights[i, 1]), 1, color='red', alpha=0.8, zorder=3)
            plt.gca().add_patch(circle)

    for i in range(agents_future_masked.shape[0]):
        future_from_current = agents_future_masked[i, current_timestep:]
        valid_mask = (
            (future_from_current[:, 0] != 0)
            & (future_from_current[:, 0] != -1)
            & (np.abs(future_from_current[:, 0]) < COORD_LIMIT)
            & (np.abs(future_from_current[:, 1]) < COORD_LIMIT)
        )

        if np.sum(valid_mask) > 1:
            valid_future = future_from_current[valid_mask]
            plt.plot(
                valid_future[:, 0],
                valid_future[:, 1],
                '--',
                color='red' if i == 0 else 'orange',
                alpha=0.6,
                linewidth=1,
            )

    plt.axis('equal')
    if debug:
        print(f"Drew {drawn_boxes} bounding boxes")






def create_trajectory_video(
    pkl_data: dict,
    pred_trajs: List[Tuple[str, np.ndarray]],
    save_path: str | None,
    scenario_name: str,
    conditioning_futures: Dict[str, np.ndarray] | None = None,
) -> List[np.ndarray]:
    """Render GT and one or more predictions in a modern dark-themed layout."""
    original_style = plt.rcParams['axes.prop_cycle']
    plt.style.use('dark_background')

    palette = {
        'map': '#424242',
        'gt_future': '#FFAB91',
        'pred_future': '#80CBC4',
        'pred_future_alt': '#B39DDB',
        'condition_future': '#FFE082',
        'ego_box': '#FFD54F',
        'others_box': '#F06292',
    }

    agents_history = pkl_data['agents_history']
    agents_future = pkl_data['agents_future']
    polylines = pkl_data['polylines']
    traffic_lights = pkl_data['traffic_light_points']

    def make_future_tensor(pred_traj: np.ndarray) -> np.ndarray:
        future = -np.ones_like(agents_future)
        if pred_traj.shape[0] == 0:
            return future
        n_pred_agents = min(pred_traj.shape[0], N_SIM_AGENTS)
        n_pred_timesteps = min(pred_traj.shape[1], 80)
        valid_pred = pred_traj[:n_pred_agents, :n_pred_timesteps, :3]
        coord_mask = (
            (np.abs(valid_pred[:, :, 0]) < COORD_LIMIT)
            & (np.abs(valid_pred[:, :, 1]) < COORD_LIMIT)
        )
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, :3] = valid_pred
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, 0][~coord_mask] = -1
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, 1][~coord_mask] = -1

        if n_pred_timesteps > 1:
            dt = 0.1
            pos_diff = np.diff(valid_pred[:, :, :2], axis=1)
            vel = pos_diff / dt
            vel_mask = np.abs(vel) < 100
            filtered_vel = np.where(vel_mask, vel, 0)
            future[:n_pred_agents, 2 : min(n_pred_timesteps + 1, vel.shape[1] + 2), 3:5] = (
                filtered_vel[:, : min(n_pred_timesteps - 1, vel.shape[1])]
            )
        return future

    pred_futures = [(title, make_future_tensor(traj)) for title, traj in pred_trajs]

    images: List[np.ndarray] = []
    total_timesteps = agents_future.shape[1]
    ego_pos = agents_future[0, :, :2]
    ego_valid = (
        (ego_pos[:, 0] != 0)
        & (ego_pos[:, 0] != -1)
        & (np.abs(ego_pos[:, 0]) < COORD_LIMIT)
        & (np.abs(ego_pos[:, 1]) < COORD_LIMIT)
    )

    center_x = 0.0
    center_y = 0.0
    if np.any(ego_valid):
        first_valid = np.where(ego_valid)[0][0]
        center_x, center_y = ego_pos[first_valid]

    def plot_frame(
        ax: plt.Axes,
        future_full: np.ndarray,
        future_masked: np.ndarray,
        title: str,
        future_color: str,
        timestep: int,
        conditioning_future: np.ndarray | None = None,
    ) -> None:
        ax.set_facecolor('#111111')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(center_x - VIEW_RANGE, center_x + VIEW_RANGE)
        ax.set_ylim(center_y - VIEW_RANGE, center_y + VIEW_RANGE)
        ax.set_title(title, fontsize=12, pad=12, color='#ECEFF1')
        ax.set_xlabel('X (m)', color='#B0BEC5')
        ax.set_ylabel('Y (m)', color='#B0BEC5')
        ax.tick_params(colors='#90A4AE', labelsize=8)
        ax.grid(color='#263238', linestyle='--', linewidth=0.4, alpha=0.6)

        for poly in polylines:
            if poly[0, 0] != 0:
                valid = (
                    (poly[:, 0] != 0)
                    & (np.abs(poly[:, 0]) < COORD_LIMIT)
                    & (np.abs(poly[:, 1]) < COORD_LIMIT)
                )
                if np.any(valid):
                    ax.plot(
                        poly[valid][:, 0],
                        poly[valid][:, 1],
                        color=palette['map'],
                        linewidth=0.6,
                        alpha=0.45,
                    )

        num_agents = min(agents_history.shape[0], future_full.shape[0])

        for i in range(num_agents):
            if timestep < future_full.shape[1]:
                curr = future_full[i, timestep]
                if (
                    curr[0] != 0
                    and curr[0] != -1
                    and np.abs(curr[0]) < COORD_LIMIT
                    and np.abs(curr[1]) < COORD_LIMIT
                ):
                    heading = curr[2] if len(curr) > 2 and curr[2] != 0 else agents_history[i, -1, 2]
                    rect = plt.Rectangle(
                        (curr[0] - agents_history[i, -1, 5] / 2, curr[1] - agents_history[i, -1, 6] / 2),
                        agents_history[i, -1, 5],
                        agents_history[i, -1, 6],
                        linewidth=2,
                        color=palette['ego_box'] if i == 0 else palette['others_box'],
                        alpha=0.9 if i == 0 else 0.5,
                        zorder=5,
                        transform=transforms.Affine2D().rotate_around(curr[0], curr[1], heading) + ax.transData,
                    )
                    ax.add_patch(rect)

            future = future_masked[i, timestep:]
            valid_future = (
                (future[:, 0] != 0)
                & (future[:, 0] != -1)
                & (np.abs(future[:, 0]) < COORD_LIMIT)
                & (np.abs(future[:, 1]) < COORD_LIMIT)
            )
            if np.sum(valid_future) > 1:
                seg = future[valid_future]
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=future_color,
                    linewidth=2.2,
                    alpha=0.95,
                    linestyle='--',
                )

            if (
                conditioning_future is not None
                and i < conditioning_future.shape[0]
                and timestep < conditioning_future.shape[1]
            ):
                cond_future = conditioning_future[i, timestep:]
                valid_cond = (
                    (cond_future[:, 0] != 0)
                    & (cond_future[:, 0] != -1)
                    & (np.abs(cond_future[:, 0]) < COORD_LIMIT)
                    & (np.abs(cond_future[:, 1]) < COORD_LIMIT)
                )
                if np.sum(valid_cond) > 1:
                    cond_seg = cond_future[valid_cond]
                    ax.plot(
                        cond_seg[:, 0],
                        cond_seg[:, 1],
                        color=palette['condition_future'],
                        linewidth=2.0,
                        alpha=0.7,
                        linestyle='-',
                    )

        ax.scatter(
            traffic_lights[:, 0],
            traffic_lights[:, 1],
            c='#FF5252',
            s=20,
            alpha=0.85,
            edgecolors='none',
            zorder=4,
        )

    panel_titles: List[Tuple[str, np.ndarray, str]] = [
        (f"{scenario_name} — Ground Truth", agents_future, palette['gt_future'])
    ]
    alt_colors = [
        palette['pred_future'],
        palette['pred_future_alt'],
        palette['condition_future'],
        '#4DD0E1',
    ]
    for idx, (title, future) in enumerate(pred_futures):
        panel_titles.append((title, future, alt_colors[idx % len(alt_colors)]))

    for t in range(0, total_timesteps, 2):
        if np.any(ego_valid):
            idx = min(t, len(ego_pos) - 1)
            while idx >= 0 and not ego_valid[idx]:
                idx -= 1
            if idx >= 0:
                center_x, center_y = ego_pos[idx]

        fig, axes = plt.subplots(1, len(panel_titles), figsize=(6 * len(panel_titles), 6))
        axes = np.atleast_1d(axes)

        gt_partial = agents_future.copy()
        if t + 1 < gt_partial.shape[1]:
            gt_partial[:, t + 1 :] = -1

        for ax, (title, future_full, color) in zip(axes, panel_titles):
            partial = future_full.copy()
            if t + 1 < partial.shape[1]:
                partial[:, t + 1 :] = -1
            conditioning_future = None
            if conditioning_futures is not None:
                conditioning_future = conditioning_futures.get(title)
            plot_frame(
                ax,
                future_full,
                partial,
                f"{title} · t={t * 0.1:.1f}s",
                color,
                t,
                conditioning_future=conditioning_future,
            )

        fig.tight_layout(pad=1.2)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img[:, :, :3].copy())
        plt.close(fig)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with mediapy.set_show_save_dir(os.path.dirname(save_path)):
            mediapy.show_video(images, title=scenario_name, fps=10)
        print(f"Saved video to {os.path.dirname(save_path)}/{scenario_name}.mp4")

    plt.rcParams['axes.prop_cycle'] = original_style
    plt.style.use('default')
    return images

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_openloop_metrics(
    gt_future: np.ndarray,
    pred_traj: np.ndarray,
    interested_mask: np.ndarray,
) -> Tuple[float, float, int, int]:
    """Return ADE, FDE, number of valid agents, and number of skipped agents."""
    n_agents = min(gt_future.shape[0], pred_traj.shape[0], N_SIM_AGENTS)
    n_timesteps = min(pred_traj.shape[1], gt_future.shape[1] - 1, 80)

    gt_pos = gt_future[:n_agents, 1 : 1 + n_timesteps, :2]
    pred_pos = pred_traj[:n_agents, :n_timesteps, :2]
    mask = interested_mask[:n_agents]

    ade_per_agent: List[float] = []
    fde_per_agent: List[float] = []
    skipped_agents = 0

    for i in range(n_agents):
        if not mask[i]:
            continue

        gt_valid = (
            (gt_pos[i, :, 0] != 0)
            & (gt_pos[i, :, 0] != -1)
            & (np.abs(gt_pos[i, :, 0]) < COORD_LIMIT)
            & (np.abs(gt_pos[i, :, 1]) < COORD_LIMIT)
        )
        pred_valid = (
            (pred_pos[i, :, 0] != 0)
            & (pred_pos[i, :, 0] != -1)
            & (np.abs(pred_pos[i, :, 0]) < COORD_LIMIT)
            & (np.abs(pred_pos[i, :, 1]) < COORD_LIMIT)
        )

        both_valid = gt_valid & pred_valid
        if not np.any(both_valid):
            skipped_agents += 1
            continue

        valid_gt = gt_pos[i, both_valid]
        valid_pred = pred_pos[i, both_valid]
        distances = np.linalg.norm(valid_gt - valid_pred, axis=1)
        ade_per_agent.append(float(np.mean(distances)))

        last_valid_idx = np.where(both_valid)[0][-1]
        final_error = np.linalg.norm(gt_pos[i, last_valid_idx] - pred_pos[i, last_valid_idx])
        fde_per_agent.append(float(final_error))

    ade = float(np.mean(ade_per_agent)) if ade_per_agent else float('nan')
    fde = float(np.mean(fde_per_agent)) if fde_per_agent else float('nan')
    return ade, fde, len(ade_per_agent), skipped_agents

# -----------------------------------------------------------------------------
# Evaluation driver
# -----------------------------------------------------------------------------




def run_simulation(args: argparse.Namespace) -> None:
    if args.model_path is None:
        raise ValueError("--model_path must be provided")
    if args.test_path is None:
        raise ValueError("--test_path must be provided")
    if not args.use_training_format:
        raise NotImplementedError(
            "TFRecord closed-loop testing is no longer supported; "
            "please export PKL files and pass --use_training_format."
        )

    base_model = VBDTest.load_from_checkpoint(args.model_path, args.device)
    base_model.reset_agent_length(N_SIM_AGENTS)
    base_model.eval()

    control_model = None
    if args.ctrl_ckpt_path:
        control_model = VBDCtrlTest.load_from_checkpoint(args.ctrl_ckpt_path, args.device)
        control_model.reset_agent_length(N_SIM_AGENTS)
        control_model.eval()

    set_seed(args.seed)

    model_specs: List[Tuple[str, torch.nn.Module, str]] = [
        ('baseline', base_model, 'Baseline')
    ]
    if control_model is not None:
        model_specs.append(('controlnet', control_model, 'ControlNet'))

    pkl_files = load_pkl_files(args.test_path)
    if not pkl_files:
        raise RuntimeError(f"No PKL files found in {args.test_path}")

    if args.max_scenarios > 0:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(pkl_files))
    else:
        order = np.arange(len(pkl_files))

    save_dir = os.path.join('testing_results', f'test_{args.test_mode}', str(args.seed))
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    header = ['scenario_id']
    for name, _, _ in model_specs:
        header.extend([
            f'ADE_{name}',
            f'FDE_{name}',
            f'num_valid_{name}',
            f'num_skipped_{name}',
        ])

    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(header)

    selected_files: List[str] = []
    processed_valid = 0

    for idx in order:
        scenario_path = pkl_files[idx]
        scenario_id = os.path.basename(scenario_path).replace('.pkl', '')
        print(f"Running scenario {scenario_id}...")

        try:
            scenario_raw = load_pkl_data(scenario_path) 
            scenario_gt = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in scenario_raw.items() # 複製一份 for gt 
            }
            batch = prepare_model_batch(scenario_raw)
            override_future_with_forward_step(batch, distance=5.0)

            # After batch creation we overwrite its future with the simple sketch; this batch is
            # only used for model conditioning. Ground-truth copies stay in `scenario_gt` above.
            conditioning_futures: dict[str, np.ndarray] = {}
            if 'agents_future' in batch:
                conditioning_futures['ControlNet'] = (
                    batch['agents_future'][0].detach().cpu().numpy().copy()
                )

            # Clone every tensor so baseline and ControlNet operate on independent inputs.
            # `batch` remains our mutable source (with the sketch); `base_batch` holds detached
            # copies that each model will clone again before inference.
            base_batch = {
                key: value.clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }

            metrics_row = [scenario_id]
            predictions_for_vis: List[Tuple[str, np.ndarray]] = []
            baseline_valid = False
            controlnet_present = False

            for name, model, title in model_specs:
                # Each model receives its own clone to guard against in-place ops performed
                # during forward passes (e.g. Lightning moving tensors to device).
                model_inputs = {
                    key: value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in base_batch.items()
                }
                with torch.no_grad():
                    if args.test_mode == 'diffusion':
                        pred = model.sample_denoiser(model_inputs)
                        pred_traj = pred['denoised_trajs'].cpu().numpy()[0]
                    elif args.test_mode == 'prior':
                        pred = model.inference_predictor(model_inputs)
                        scores = pred['goal_scores'][0].softmax(dim=-1)
                        trajs = pred['goal_trajs'][0]
                        sampled_idx = torch.multinomial(scores, 1).squeeze(-1)
                        agent_idx = torch.arange(trajs.shape[0], device=sampled_idx.device)
                        pred_traj = trajs[agent_idx, sampled_idx].cpu().numpy()
                    else:
                        raise NotImplementedError(f"Unsupported test_mode: {args.test_mode}")

                gt_future = scenario_gt['agents_future']
                interested_mask = scenario_gt['agents_interested'] > 0
                ade, fde, num_valid, num_skipped = compute_openloop_metrics(
                    gt_future, pred_traj, interested_mask
                )

                if name == 'baseline' and num_valid > 0:
                    baseline_valid = True

                metrics_row.extend([ade, fde, num_valid, num_skipped])

                if num_valid == 0:
                    print(
                        f"  ⚠️ {title}: no overlapping valid timesteps; metrics reported as NaN."
                    )
                else:
                    print(
                        f"  {title}: ADE={ade:.3f}, FDE={fde:.3f}, valid={num_valid}, skipped={num_skipped}"
                    )

                predictions_for_vis.append((title, pred_traj))

                if name == 'controlnet':
                    controlnet_present = True

                if name == 'controlnet' and 'ControlNet' in conditioning_futures:
                    conditioning_futures[title] = conditioning_futures.pop('ControlNet')


            if not baseline_valid:
                print("  ⚠️ Baseline invalid for this scenario; skipping.")
                continue

            with open(metrics_path, 'a', newline='') as f:
                csv.writer(f).writerow(metrics_row)

            selected_files.append(scenario_path)
            processed_valid += 1

            if args.save_simulation:
                vis_path = os.path.join(save_dir, f'{scenario_id}_video')
                create_trajectory_video(
                    scenario_gt,
                    predictions_for_vis,
                    vis_path,
                    scenario_id,
                    conditioning_futures=(
                        conditioning_futures
                        if controlnet_present and conditioning_futures
                        else None
                    ),
                )

            if args.max_scenarios > 0 and processed_valid >= args.max_scenarios:
                break

        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"Error processing {scenario_id}: {exc}")
            continue

    if args.max_scenarios > 0 and processed_valid < args.max_scenarios:
        print(
            f"⚠️ Requested {args.max_scenarios} scenarios but only {processed_valid} valid ones were found."
        )

    if selected_files:
        print(
            "Selected scenarios (seed {}): {}".format(
                args.seed,
                [os.path.basename(f) for f in selected_files],
            )
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, required=True, help='Directory containing processed PKL files')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_mode', type=str, default='diffusion', choices=['diffusion', 'prior'])
    parser.add_argument('--save_simulation', action='store_true')
    parser.add_argument('--use_training_format', action='store_true', help='Must be set; TFRecord format is not supported')
    parser.add_argument('--max_scenarios', type=int, default=-1, help='Number of valid scenarios to evaluate (-1 for all)')
    parser.add_argument('--ctrl_ckpt_path', type=str, default=None, help='Optional ControlNet checkpoint for side-by-side comparison')
    return parser.parse_args()

if __name__ == '__main__':
    cli_args = parse_args()
    run_simulation(cli_args)
