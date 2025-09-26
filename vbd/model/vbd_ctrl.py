import torch
import torch.nn as nn
import lightning.pytorch as pl
from .modules import Encoder, Denoiser, GoalPredictor, Ctrl_Denoiser, Ctrl_TransformerDecoder
from .utils import DDPM_Sampler
from .model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import smooth_l1_loss, cross_entropy


class FutureEncoder(nn.Module):
    """
    Simple MLP-based encoder for future agent trajectories.
    Replaces complex conv layers to eliminate numerical instability.
    """
    def __init__(self, input_dim=3, feature_dim=128, agents_len=32):
        super().__init__()
        self.agents_len = agents_len
        self.feature_dim = feature_dim

        # Simple MLP approach: flatten temporal dimension and process with MLPs
        # For future_len=81, input_dim=3: total input = 81*3 = 243
        self.temporal_len = 81  # From config
        self.total_input_dim = self.temporal_len * input_dim  # 81*3 = 243 (x,y,heading * 81 Timesteps)

        self.input_norm = nn.LayerNorm(self.total_input_dim)  # norm for gradient stability

        # Very simple MLP layers
        encoder_layers = [
            nn.Linear(self.total_input_dim, 256),
            nn.ReLU(),
        ]
        encoder_layers.append(nn.Linear(256, feature_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Initialize with smaller weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Conservative gain, start from nearly 0 
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, agents_future):
        """
        Args:
            agents_future: [B, A, T, C] - future agent trajectories
        Returns:
            future_features: [B, A, F] - encoded future features
        """
        B, A, T, C = agents_future.shape # C should be 5 or 8, first 3 is x, y, heading

        # Take only the first 3 channels (x, y, heading)
        agents_future = agents_future[..., :3]  # [B, A, T, 3] 

        # Convert to relative coordinates (relative to initial position)
        # Initial position is at t=0
        initial_pos = agents_future[:, :, 0:1, :]  # [B, A, 1, 3]

        # For x, y coordinates: subtract initial position
        relative_future = agents_future.clone()
        relative_future[..., :2] = agents_future[..., :2] - initial_pos[..., :2]  # [B, A, T, 2]

        # For heading: compute relative heading (difference from initial heading)
        relative_future[..., 2] = agents_future[..., 2] - initial_pos[..., 2]  # [B, A, T]

        # Normalize heading differences to [-π, π] range
        relative_future[..., 2] = torch.atan2(
            torch.sin(relative_future[..., 2]),
            torch.cos(relative_future[..., 2])
        )

        # Flatten temporal dimension: [B, A, T*3]
        flattened = relative_future.reshape(B, A, -1)  # [B, A, 243]

        # Apply MLP to each agent independently
        # Reshape to [B*A, 243], process, then reshape back
        x = flattened.reshape(B*A, -1)  # [B*A, 243]
        x = self.input_norm(x)

        # Process through MLP (no normalization needed with relative coordinates)
        features = self.encoder(x)  # [B*A, feature_dim]

        # Reshape back to [B, A, feature_dim]
        future_features = features.reshape(B, A, self.feature_dim)  # [B, A, 128]

        return future_features


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer for conditioning.
    """
    def __init__(self, feature_dim, conditioning_dim, gamma_scale=2.0, beta_scale=4.0):
        super().__init__()
        self.feature_dim = feature_dim # feature is the query from denoiser
        self.conditioning_dim = conditioning_dim
        self.gamma_scale = gamma_scale # gamma and beta is for stabilization
        self.beta_scale = beta_scale
        self._last_stats = {}

        # gamma_raw、beta_raw 就是「conditioning 經過 MLP 後」得到的新的兩個向量，各自對應 denoiser 特徵的每一個 channel
        mlp_layers = [
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(),
        ]
        mlp_layers.append(nn.Linear(conditioning_dim, 2 * feature_dim))  # gamma and beta (raw)
        self.film_mlp = nn.Sequential(*mlp_layers)

        # Initialize to identity transformation
        self._init_weights()

    def _init_weights(self):
        # Initialize so that gamma starts at 1 and beta starts at 0
        with torch.no_grad():
            self.film_mlp[-1].weight.fill_(0.0)
            self.film_mlp[-1].bias[:self.feature_dim].fill_(1.0)  # gamma = 1
            self.film_mlp[-1].bias[self.feature_dim:].fill_(0.0)  # beta = 0

    def forward(self, x, conditioning):
        """
        Args:
            x: [B, A, T, F] - denoiser query input features
            conditioning: [B, A, C] - conditioning features
        Returns:
            modulated_x: [B, A, T, F] - FiLM modulated features
        """
        B, A, T, F = x.shape


        # Expand conditioning to match temporal dimension
        conditioning_expanded = conditioning.unsqueeze(2).expand(B, A, T, -1)  # [B, A, T, C]
        conditioning_flat = conditioning_expanded.reshape(B*A*T, -1)

        # Generate gamma and beta
        film_params = self.film_mlp(conditioning_flat)  # [B*A*T, 2*F]
        gamma_raw, beta_raw = film_params.split(F, dim=1)  # Each [B*A*T, F]

        # Reshape back
        gamma_raw = gamma_raw.reshape(B, A, T, F)
        beta_raw = beta_raw.reshape(B, A, T, F)

        # Stabilize gamma/beta with smooth saturations around identity
        # gamma_raw, beta_raw start at zero -> gamma=1, beta=0
        gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)

        with torch.no_grad():
            gamma_detached = gamma.detach()
            beta_detached = beta.detach()
            self._last_stats = {
                'gamma_mean': gamma_detached.mean().item(),
                'gamma_std': gamma_detached.std(unbiased=False).item(),
                'gamma_min': gamma_detached.min().item(),
                'gamma_max': gamma_detached.max().item(),
                'beta_mean': beta_detached.mean().item(),
                'beta_std': beta_detached.std(unbiased=False).item(),
                'beta_min': beta_detached.min().item(),
                'beta_max': beta_detached.max().item(),
                'gamma_residual_mean': (gamma_detached - 1.0).abs().mean().item(),
                'beta_abs_mean': beta_detached.abs().mean().item(),
            }

        # Check for NaN/inf in FiLM parameters (DEBUG)
        if torch.isnan(gamma).any() or torch.isinf(gamma).any():
            print("WARNING: NaN/inf detected in gamma, using identity")
            gamma = torch.ones_like(gamma)
        if torch.isnan(beta).any() or torch.isinf(beta).any():
            print("WARNING: NaN/inf detected in beta, using zeros")
            beta = torch.zeros_like(beta)

        # Apply FiLM: γ(c) ⊙ x + β(c)
        modulated_x = gamma * x + beta

        return modulated_x

    @property
    def last_stats(self):
        return self._last_stats


class ControlNetModule(nn.Module):
    """
    ControlNet module that provides FiLM conditioning to the original denoiser.
    This module is trainable while the original model is frozen.
    """
    def __init__(self, future_feature_dim=128, agents_len=32, film_blocks=[0, 1],
                 film_gamma_scale=2.0, film_beta_scale=4.0):
        super().__init__()
        self._future_feature_dim = future_feature_dim
        self._agents_len = agents_len
        self._film_blocks = film_blocks
        self._film_gamma_scale = film_gamma_scale
        self._film_beta_scale = film_beta_scale

        # FutureEncoder: encodes agents_future into conditioning features
        self.future_encoder = FutureEncoder(
            input_dim=3,  # [x, y, theta]
            feature_dim=future_feature_dim,
            agents_len=agents_len
        )

        # FiLM layers for different blocks
        # We need to know the feature dimensions at each block to create FiLM layers
        # Based on the TransformerDecoder, the feature dimension is 256
        self.film_layers = nn.ModuleDict()
        for block_idx in film_blocks:
            self.film_layers[f'block_{block_idx}'] = FiLMLayer(
                feature_dim=256,  # TransformerDecoder uses 256 as feature dim
                conditioning_dim=future_feature_dim,
                gamma_scale=self._film_gamma_scale,
                beta_scale=self._film_beta_scale
            )
        self._last_stats = {}

    def encode_future(self, agents_future):
        """
        Encode future trajectories into conditioning features.

        Args:
            agents_future: [B, A, T, 3] - future agent trajectories

        Returns:
            conditioning_features: [B, A, F] - encoded conditioning features
        """
        return self.future_encoder(agents_future)

    def apply_film_conditioning(self, features, conditioning_features, block_idx):
        """
        Apply FiLM conditioning to features at a specific block.

        Args:
            features: [B, A, T, 256] - features to be modulated
            conditioning_features: [B, A, F] - conditioning features
            block_idx: int - which block this is

        Returns:
            modulated_features: [B, A, T, 256] - FiLM modulated features
        """
        if f'block_{block_idx}' in self.film_layers:
            return self.film_layers[f'block_{block_idx}'](features, conditioning_features)
        else:
            return features  # No conditioning for this block

    def get_film_layers(self):
        return {name: layer for name, layer in self.film_layers.items()}

    def get_film_stats(self):
        stats = {}
        for name, layer in self.film_layers.items():
            layer_stats = getattr(layer, 'last_stats', None)
            if layer_stats:
                stats[name] = layer_stats
        return stats
 


class VBD(pl.LightningModule):
    """
    Versertile Behavior Diffusion model.
    """

    def __init__(
        self,
        cfg: dict,
    ):
        """
        Initialize the VBD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self._future_len = cfg['future_len']
        self._agents_len = cfg['agents_len']
        self._action_len = cfg['action_len']
        self._diffusion_steps = cfg['diffusion_steps']
        self._encoder_layers = cfg['encoder_layers']
        self._encoder_version = cfg.get('encoder_version', 'v1')
        self._action_mean = cfg['action_mean']
        self._action_std = cfg['action_std']
        
        self._train_encoder = cfg.get('train_encoder', True)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', True)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._replay_buffer = cfg.get('replay_buffer', False)
        self._embeding_dim = cfg.get('embeding_dim', 5) # By default, the embed is the noised trajectory so the dimension is 5

        # ControlNet specific parameters
        self._use_controlnet = cfg.get('use_controlnet', True)
        self._future_feature_dim = cfg.get('future_feature_dim', 128)
        self._film_blocks = cfg.get('film_blocks', [0, 1])  # Which denoiser blocks to apply FiLM to
        self._film_gamma_scale = cfg.get('film_gamma_scale', 2.0)
        self._film_beta_scale = cfg.get('film_beta_scale', 4.0)
        self._freeze_original = cfg.get('freeze_original', True)  # Freeze original model weights
        # ControlNet conditioning configuration
        # Randomly drop a portion of agent futures during training to simulate missing conditioning
        self._per_agent_dropout_min = cfg.get('per_agent_dropout_min', 0.0)
        self._per_agent_dropout_max = cfg.get('per_agent_dropout_max', 0.0)
        # Optional interval (in training steps) to log guided/unguided metrics without heavy overhead every step.
        self._control_metric_interval = 0  # disable train-time guided/unguided logging by default

        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)

        # Use ControlNet-enabled denoiser
        from .modules import Ctrl_Denoiser
        self.denoiser = Ctrl_Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
            input_dim=self._embeding_dim,
        )

        # ControlNet module (trainable)
        if self._use_controlnet:
            self.controlnet = ControlNetModule(
                future_feature_dim=self._future_feature_dim,
                agents_len=self._agents_len,
                film_blocks=self._film_blocks,
                film_gamma_scale=self._film_gamma_scale,
                film_beta_scale=self._film_beta_scale,
            )
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s = cfg.get('schedule_s', 0.0),
            e = cfg.get('schedule_e', 1.0),
            tau = cfg.get('schedule_tau', 1.0),
            scale = cfg.get('schedule_scale', 1.0),
        )
                
        self.register_buffer('action_mean', torch.tensor(self._action_mean))  
        self.register_buffer('action_std', torch.tensor(self._action_std))
    
    ################### Training Setup ###################
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        # ControlNet training strategy: freeze original model, train only ControlNet
        if self._use_controlnet and self._freeze_original:
            # Freeze all original model parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.denoiser.parameters():
                param.requires_grad = False
            if self._with_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

            # Only ControlNet module is trainable
            params_to_update = list(self.controlnet.parameters())
            print(f"ControlNet mode: Training only {len(params_to_update)} ControlNet parameters")

        else:
            # Original training strategy
            if not self._train_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            if not self._train_denoiser:
                for param in self.denoiser.parameters():
                    param.requires_grad = False
            if self._with_predictor and (not self._train_predictor):
                for param in self.predictor.parameters():
                    param.requires_grad = False

            params_to_update = []
            for param in self.parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        # Use extremely small learning rate for ControlNet to prevent weight corruption
        controlnet_lr = 1e-5 if (self._use_controlnet and self._freeze_original) else self.cfg['lr']
        print(f"Using learning rate: {controlnet_lr} (ControlNet mode: {self._use_controlnet and self._freeze_original})")

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=controlnet_lr,
            weight_decay=self.cfg['weight_decay']
        )
        
        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
        
            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1
        
            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, 
                lr_warmpup_step, 
                lr_step_freq,
                lr_step_gamma,
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the VBD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary containing the model outputs.
        """
        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)
        
        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(encoder_outputs, noised_actions_normalized, diffusion_step)
            output_dict.update(denoiser_outputs)
            
        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)
            
        return output_dict
        
    def forward_denoiser(
        self,
        encoder_outputs,
        noised_actions_normalized,
        diffusion_step,
        agents_future=None,
        control_mask=None,
    ):
        """
        Forward pass of the denoiser module with optional ControlNet conditioning.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.
            agents_future: [B, A, T, 3] - future trajectories for ControlNet conditioning (optional).

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """
        noised_actions = self.unnormalize_actions(noised_actions_normalized)

        # Get ControlNet conditioning if available
        future_conditioning = None
        film_layers = None
        if self._use_controlnet and agents_future is not None:
            future_conditioning = self.controlnet.encode_future(agents_future)
            film_layers = self.controlnet.get_film_layers()

        # Forward through ControlNet-enabled denoiser
        denoiser_output = self.denoiser(
            encoder_outputs, noised_actions, diffusion_step,
            future_conditioning=future_conditioning,
            film_layers=film_layers,
            control_mask=control_mask,
        )

        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_step,
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'
        
        # Roll out
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_trajs = roll_out(current_states, denoised_actions,
                    action_len=self.denoiser._action_len, global_frame=True)
        
        return {
            'denoiser_output': denoiser_output,
            'denoised_actions_normalized': denoised_actions_normalized,
            'denoised_actions': denoised_actions,
            'denoised_trajs': denoised_trajs,
        }
    
    def forward_predictor(self, encoder_outputs):
        """
        Forward pass of the predictor module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Predict goal
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        goal_actions = self.unnormalize_actions(goal_actions_normalized)    
        goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
                    action_len=self.predictor._action_len, global_frame=True)
        
        return {
            'goal_actions_normalized': goal_actions_normalized,
            'goal_actions': goal_actions,
            'goal_scores': goal_scores,
            'goal_trajs': goal_trajs,
        }
        
    def forward_and_get_loss(self, batch, prefix = '', debug = False):
        """
        Forward pass of the model and compute the loss.

        Args:
            batch: Input batch.
            prefix: Prefix for the loss keys.
            debug: Flag to enable debug mode.

        Returns:
            total_loss: Total loss.
            log_dict: Dictionary containing the loss values.
            debug_outputs: Dictionary containing debug outputs.
        """
        # data inputs
        agents_future = batch['agents_future'][:, :self._agents_len]
        
        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch['agents_interested'][:, :self._agents_len]
        anchors = batch['anchors'][:, :self._agents_len]

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        control_agents_future = agents_future
        control_dropout_mask = None
        # Base mask: agents that both interest us and have valid future trajectories
        valid_agents = (agents_interested > 0)
        valid_agents = valid_agents & agents_future_valid.any(dim=-1)
        control_mask = valid_agents.float()
        if (
            self.training
            and self._use_controlnet
            and prefix.startswith('train/')
            and self._per_agent_dropout_max > 0.0
        ):
            # Sample a dropout probability for this batch within the configured range
            dropout_prob = torch.empty(1, device=agents_future.device).uniform_(
                self._per_agent_dropout_min, self._per_agent_dropout_max
            ).item()
            dropout_prob = float(max(0.0, min(1.0, dropout_prob)))

            if dropout_prob > 0.0:
                if valid_agents.any():
                    control_agents_future = agents_future.clone()
                    random_tensor = torch.rand_like(valid_agents, dtype=torch.float32)
                    # Drop per-agent conditioning by zeroing their future trajectories
                    control_dropout_mask = (random_tensor < dropout_prob) & valid_agents

                    for b in range(control_dropout_mask.shape[0]):
                        valid_mask_b = valid_agents[b]
                        if valid_mask_b.any():
                            dropped_b = control_dropout_mask[b] & valid_mask_b
                            if dropped_b.all():
                                # Ensure at least one conditioned agent remains per batch
                                keep_idx = torch.nonzero(valid_mask_b, as_tuple=False)[0]
                                control_dropout_mask[b, keep_idx] = False

                    dropout_expanded = control_dropout_mask.unsqueeze(-1).unsqueeze(-1)
                    dropout_expanded = dropout_expanded.expand_as(control_agents_future)
                    # Zero the futures for dropped agents (preserving tensor shape)
                    control_agents_future = control_agents_future.masked_fill(dropout_expanded, 0.0)

                    dropped = (control_dropout_mask & valid_agents).sum().item()
                    total_valid = valid_agents.sum().item()
                    dropout_ratio = dropped / total_valid if total_valid > 0 else 0.0

                    log_dict[f'{prefix}control_dropout_ratio'] = dropout_ratio

                    # Expose mask for debugging / visualization
                    debug_outputs['control_dropout_mask'] = control_dropout_mask.detach().clone()

                    # Mask marks agents with available conditioning (1) vs dropped (0)
                    control_mask = (~control_dropout_mask & valid_agents).float()
                else:
                    control_mask.zero_()
            # If dropout prob resolves to 0.0, keep the default control_mask (valid_agents)

        # get actions from trajectory
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len,
        )
        
        gt_actions_normalized = self.normalize_actions(gt_actions)
        B, A, T, D = gt_actions_normalized.shape

        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        
        ############### Denoise #################
        if self._train_denoiser:
            
            diffusion_steps = torch.randint(
                0, self.noise_scheduler.num_steps, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)
            
            # sample noise 
            # noise = torch.randn(B*A, T, D).type_as(agents_future)
            noise = torch.randn(B, A, T, D).type_as(agents_future)
            
            # noise the input
            noised_action_normalized = self.noise_scheduler.add_noise(
                gt_actions_normalized, #.reshape(B*A, T, D),
                noise,
                diffusion_steps#, .reshape(B*A),
            )#.reshape(B, A, T, D)
            # noise = noise.reshape(B, A, T, D)

            if self._replay_buffer:
                with torch.no_grad():
                    # Forward for one step
                    denoise_outputs = self.forward_denoiser(
                        encoder_outputs, gt_actions_normalized, diffusion_steps.view(B,A),
                        agents_future=agents_future if self._use_controlnet else None,
                        control_mask=control_mask if self._use_controlnet else None,
                    )
                    
                    x_0 = denoise_outputs['denoised_actions_normalized']
        
                    # Step to sample from P(x_t-1 | x_t, x_0)
                    x_t_prev = self.noise_scheduler.step(
                        model_output = x_0,
                        timesteps = diffusion_steps,
                        sample = noised_action_normalized,
                        prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
                    )
                    noised_action_normalized = x_t_prev.detach()
            
            # Feed the possibly masked future tensor into the ControlNet-conditioned denoiser
            denoise_outputs = self.forward_denoiser(
                encoder_outputs,
                noised_action_normalized,
                diffusion_steps.view(B, A),
                agents_future=control_agents_future if self._use_controlnet else None,
                control_mask=control_mask if self._use_controlnet else None,
            )
            
            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            if self._use_controlnet:
                film_stats = self.controlnet.get_film_stats()
                for block_name, stats in film_stats.items():
                    for key, value in stats.items():
                        log_dict[f"{prefix}film_{block_name}_{key}"] = value

            # Get Loss 
            denoised_trajs = denoise_outputs['denoised_trajs']
            if self._prediction_type == 'sample':
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )
                denoise_loss = state_loss_mean + yaw_loss_mean 
                total_loss += denoise_loss
                
                # Predict the noise
                _, diffusion_loss = self.noise_scheduler.get_noise(
                    x_0 = denoise_outputs['denoised_actions_normalized'],
                    x_t = noised_action_normalized,
                    timesteps=diffusion_steps,
                    gt_noise=noise,
                )
                                
                log_dict.update({
                    prefix+'state_loss': state_loss_mean.item(),
                    prefix+'yaw_loss': yaw_loss_mean.item(),
                    prefix+'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean':
                pred_action_normalized = denoise_outputs['denoised_actions_normalized']
                denoise_loss = self.action_loss(
                    pred_action_normalized, gt_actions_normalized, gt_actions_valid, agents_interested
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'action_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')
                

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            if self._use_controlnet and prefix.startswith('val/'):
                guided_mask = control_mask > 0.5
                unguided_mask = guided_mask.logical_not()

                if guided_mask.any():
                    guided_valid = agents_future_valid & guided_mask.unsqueeze(-1)
                    guided_ade, guided_fde = self.calculate_metrics_denoise(
                        denoised_trajs, agents_future, guided_valid, agents_interested, 8
                    )
                    log_dict[f'{prefix}guided_ADE'] = guided_ade
                    log_dict[f'{prefix}guided_FDE'] = guided_fde

                if unguided_mask.any():
                    unguided_valid = agents_future_valid & unguided_mask.unsqueeze(-1)
                    unguided_ade, unguided_fde = self.calculate_metrics_denoise(
                        denoised_trajs, agents_future, unguided_valid, agents_interested, 8
                    )
                    log_dict[f'{prefix}unguided_ADE'] = unguided_ade
                    log_dict[f'{prefix}unguided_FDE'] = unguided_fde
            
            log_dict.update({
                prefix+'denoise_ADE': denoise_ade,
                prefix+'denoise_FDE': denoise_fde,
            })
        
        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss 
            goal_scores = goal_outputs['goal_scores']
            goal_trajs = goal_outputs['goal_trajs']
            
            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )

            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss += 1.0 * pred_loss 
            
            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            
            log_dict.update({
                prefix+'goal_loss': goal_loss_mean.item(),
                prefix+'score_loss': score_loss_mean.item(),
                prefix+'pred_ADE': pred_ade,
                prefix+'pred_FDE': pred_fde,
            })
        
        log_dict[prefix+'loss'] = total_loss.item()
        
        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict
    
    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """        
        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        self.log_dict(
            log_dict, 
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(log_dict, 
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)
        
        return loss

    ################### Loss function ###################
    def denoise_loss(
            self, denoised_trajs,
            agents_future, agents_future_valid,
            agents_interested
        ):
            """
            Calculates the denoise loss for the denoised actions and trajectories.

            Args:
                denoised_actions_normalized (torch.Tensor): Normalized denoised actions tensor of shape [B, A, T, C].
                denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
                agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
                agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
                gt_actions_normalized (torch.Tensor): Normalized ground truth actions tensor of shape [B, A, T, C].
                gt_actions_valid (torch.Tensor): Ground truth actions validity tensor of shape [B, A, T].
                agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].

            Returns:
                state_loss_mean (torch.Tensor): Mean state loss.
                yaw_loss_mean (torch.Tensor): Mean yaw loss.
                action_loss_mean (torch.Tensor): Mean action loss.
            """
            
            agents_future = agents_future[..., 1:, :3]
            future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

            # Calculate State Loss
            # [B, A, T]
            state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(-1)
            yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_loss = torch.abs(yaw_error)
            
            # Filter out the invalid state
            state_loss = state_loss * future_mask
            yaw_loss = yaw_loss * future_mask

            mask_sum = future_mask.sum()
            if mask_sum.item() == 0:
                zero = denoised_trajs.sum() * 0.0
                return zero, zero

            # Calculate the mean loss
            state_loss_mean = state_loss.sum() / mask_sum
            yaw_loss_mean = yaw_loss.sum() / mask_sum
            
            return state_loss_mean, yaw_loss_mean
        
    def action_loss(
        self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        
        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask

        mask_sum = action_mask.sum()
        if mask_sum.item() == 0:
            return actions.sum() * 0.0

        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / mask_sum

        return action_loss_mean
    
    def goal_loss(
        self, trajs, scores, agents_future,
        agents_future_valid, anchors,
        agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B*A, Q, T, 3] representing predicted trajectories.
            scores (torch.Tensor): Tensor of shape [B*A, Q] representing predicted scores.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            anchors (torch.Tensor): Tensor of shape [B, A, Q, 2] representing anchor points.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        # Convert Anchor to Global Frame
        current_states = agents_future[:, :, 0, :3] 
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape
        
        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0) # [B, A, T]
        
        # Flatten batch and agents
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1) # [B*A, 1, 2]
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1) # [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3] # [B*A, Q, T, 3]
        anchors_global = anchors_global.flatten(0, 1) # [B*A, Q, 2]
        
        # Find the closest anchor
        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1) # [B*A,]
        
        # For agents that do not have valid end point, use the minADE
        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1) # [B*A, Q, T]
        dist = dist * traj_mask.flatten(0, 1)[:, None, :] # [B*A, Q, T]
        idx = torch.argmin(dist.mean(-1), dim=-1) # [B*A,]

        # Select trajectory
        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch*num_agents), idx] # [B*A, T, 3]
        
        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction='none').sum(-1) # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1) # [B*A, T]

        traj_mask_sum = traj_mask.sum()
        interested_mask = (agents_interested > 0)
        interested_sum = interested_mask.sum()

        if traj_mask_sum.item() == 0 or interested_sum.item() == 0:
            zero = trajs.sum() * 0.0
            return zero, zero

        # Calculate the score loss
        scores = scores.flatten(0, 1) # [B*A, Q]
        score_loss = cross_entropy(scores, idx, reduction='none') # [B*A]
        score_loss = score_loss * interested_mask.flatten(0, 1) # [B*A]

        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask_sum
        score_loss_mean = score_loss.sum() / interested_sum

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self, 
            denoised_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the denoising metrics for the predicted trajectories.

            Args:
                denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
                agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
                agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
                agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
                top_k (int, optional): Number of top agents to consider. Defaults to None.

            Returns:
                Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
            """
            
            if not top_k:
                top_k = self._agents_len  
            
            pred_traj = denoised_trajs[:, :top_k, :, :2] # [B, A, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 

            valid_count = gt_mask.sum()
            if valid_count.item() == 0:
                return 0.0, 0.0

            denoise_mse = torch.norm(pred_traj - gt, dim = -1)
            denoise_ADE = denoise_mse[gt_mask].mean()

            final_mask = gt_mask[..., -1]
            if final_mask.sum().item() == 0:
                denoise_FDE = denoise_ADE
            else:
                denoise_FDE = denoise_mse[..., -1][final_mask].mean()
            
            return denoise_ADE.item(), denoise_FDE.item()
    
    @torch.no_grad()
    def calculate_metrics_predict(self,
            goal_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the metrics for predicting goal trajectories.

            Args:
                goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
                agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
                agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
                agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
                top_k (int, optional): The number of top agents to consider. Defaults to None.

            Returns:
                tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
            """
            
            if not top_k:
                top_k = self._agents_len
            goal_trajs = goal_trajs[:, :top_k, :, :, :2] # [B, A, Q, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 
                   
            valid_count = gt_mask.sum()
            if valid_count.item() == 0:
                return 0.0, 0.0

            goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim = -1) # [B, A, Q, T]
            goal_mse = goal_mse * gt_mask[..., None, :] # [B, A, Q, T]
            best_idx = torch.argmin(goal_mse.sum(-1), dim = -1) 
            
            best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
                                     torch.arange(goal_mse.shape[1])[None, :],
                                     best_idx]
            
            goal_ADE = best_goal_mse.sum() / valid_count
            final_mask = gt_mask[..., -1]
            if final_mask.sum().item() == 0:
                goal_FDE = goal_ADE
            else:
                goal_FDE = best_goal_mse[..., -1].sum()/final_mask.sum()
            
            return goal_ADE.item(), goal_FDE.item()
    
    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        """
        Normalize the given actions using the mean and standard deviation.

        Args:
            actions : The actions to be normalized.

        Returns:
            The normalized actions.
        """
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_actions(self, actions: torch.Tensor):
        """
        Unnormalize the given actions using the stored action standard deviation and mean.

        Args:
            actions: The normalized actions to be unnormalized.

        Returns:
             The unnormalized actions.
        """
        return actions * self.action_std + self.action_mean
    
