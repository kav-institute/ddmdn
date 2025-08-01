import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.shared_layers import ClassicMLP, ConcatSquash
from torch.nn import Module
from box import Box
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler


# ************************************
# Overview:
#
# MDN Diffusion: Section 3.4 and 3.5
#
# ************************************


class PositionalEncoding(Module):
    """
    Sinusoidal positional encoding.
    Args:
        d_model (int): Encoding dimension.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length.
        transpose (bool): If True, output shape is [max_len,1,d_model], else [1,max_len,d_model].
    """
    
    def __init__(self, d_model, dropout, max_len, transpose=False):
        """
        Precompute positional encodings and store as buffer.
        Args:
            d_model, dropout, max_len, transpose: see class doc.
        """
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if transpose: pe = pe.unsqueeze(0).transpose(0, 1)
        else: pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
        
        
    def forward(self, x):
        """
        Add positional encoding to input.
        Args:
            x (Tensor): Input tensor, shape matches pe prefix: [T, B, d_model] or [B, T, d_model].
        Returns:
            Tensor: Encoded tensor with same shape as x.
        """
        
        x = x + self.pe[: x.size(0), :]  # [T, B, d_model] or [B, T, d_model]
        return self.dropout(x)
    
    
class VarianceSchedule(Module):
    """
    Diffusion noise schedule generator.
    Args:
        num_steps (int): Number of diffusion timesteps.
        mode (str): 'linear' or 'cosine' schedule.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
        cosine_s (float): Small offset for cosine schedule.
    """
    
    def __init__(self, num_steps, mode, beta_start, beta_end, importance_sampling=False, decay=0.99, cosine_s=8e-3):
        """
        Compute betas, alphas, alpha_bars, and sigmas for schedule.
        """
        
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.mode = mode
        self.importance_sampling = importance_sampling
        
        # Schedule mode types
        if mode == 'linear':
            
            betas = torch.linspace(beta_start, beta_end, steps=num_steps)
            
        elif mode == 'cosine':
            
            timesteps = (torch.arange(num_steps + 1) / num_steps + cosine_s)
            alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
            
        # Determine parameters
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        
        for i in range(1, log_alphas.size(0)):
            
            log_alphas[i] += log_alphas[i - 1]
            
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        
        for i in range(1, sigmas_flex.size(0)):
            
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
            
        sigmas_inflex = torch.sqrt(sigmas_inflex)
        
        # Register to buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        
        # Importance sampling buffers
        if self.importance_sampling:
            
            # One extra slot since betas has length num_steps+1
            self.register_buffer("step_ema", torch.zeros(num_steps+1))
            self.decay = decay
        
        
    def uniform_sample_t(self, batch_size):
        """
        Sample timesteps uniformly between 1 and num_steps.
        Args:
            batch_size (int): Number of samples.
        Returns:
            list[int]: Random timesteps.
        """
        
        return np.random.choice(np.arange(1, self.num_steps+1), batch_size).tolist()
    
    
    def weighted_sample_t(self, batch_size):
        """
        Sample timesteps with p_t ∝ 1/sqrt(EMA[ℓ_t] + ε).
        """
        
        # Avoid zero-division
        weights = 1.0 / (torch.sqrt(self.step_ema[1:] + 1e-8))
        probs = weights / weights.sum()
        
        # Sample indices in [0, num_steps-1], then shift by +1
        idx = torch.multinomial(probs, batch_size, replacement=True)
        return (idx + 1).tolist()
    
    
    def update_step_losses(self, t_list, per_sample_losses):
        """
        t_list: list[int] of length B
        per_sample_losses: Tensor[B] containing ℓ_t for each sample
        """
        
        for t, l in zip(t_list, per_sample_losses):
            
            # EMA: new = decay*old + (1-decay)*current
            self.step_ema[t] = self.decay * self.step_ema[t] + (1 - self.decay) * l.detach()
    
    
    def get_sigmas(self, t, flexibility):
        """
        Compute noise scale (sigma) for given timesteps.
        Args:
            t (int or Tensor): Timestep index/indices.
            flexibility (float): Weight for flexible schedule.
        Returns:
            Tensor: Sigma values.
        """
        
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    
class MixtureLinear(Module):
    
    def __init__(self, cfg, dims):
        
        super().__init__()
        
        # Params
        self.T = cfg.len_forecast_horizon
        self.mixture_dim = cfg.mixture_dim
        self.context_dim = dims.mdn_context_dim
        self.variance_epsilon =cfg.model.mdn_variance_epsilon
        self.out_dim = self.T * self.mixture_dim
        self.mixture_params = None
        self.per_traj_w = None
        self.per_step_w = None
        
        # Layers
        self.mdn_params = nn.Linear(self.context_dim, dims.mdn_emb_dim)
        self.per_traj_weights = nn.Linear(self.context_dim, cfg.num_modes)
        self.per_step_weights = nn.Linear(self.context_dim, cfg.num_modes * self.T)
        
        
        # Initialize weights
        self._init_weights()
        
        
    def _init_weights(self):
        """
        Initialize linear layer weights with Xavier uniform and biases to zero.
        """
        
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    
    def forward(self, context, t=None, B=1):
        """
        Compute MDN parameters for input embeddings.
        Args:
            context (Tensor): Agent embeddings [B, input_embedding_dim].
        Returns:
            Tensor: MDN latent tensor [B, T, mixture_dim].
        """
        
        B = context.shape[0]
        self.mixture_params = self.mdn_params(context) # [B, T*mixture_dim]
        self.per_traj_w = self.per_traj_weights(context)  # [B, M]
        self.per_step_w = self.per_step_weights(context)  # [B, T*M]
        return 
    
    
    def sample(self, context=None, B=1):
        
        return self.mixture_params, self.per_traj_w, self.per_step_w
    
    
class MixtureMLP(Module):
    
    def __init__(self, cfg, dims):
        
        super().__init__()
        
        # Params
        self.T = cfg.len_forecast_horizon
        self.mixture_dim = cfg.mixture_dim
        self.context_dim = dims.mdn_context_dim
        self.variance_epsilon =cfg.model.mdn_variance_epsilon
        self.mlp_activation = cfg.model.mdn_mlp_activation
        self.mlp_hidden_dim = cfg.model.mdn_mlp_hidden_dim
        self.mlp_dropout_prob = cfg.model.mdn_mlp_dropout
        self.out_dim = cfg.model.mdn_mlp_out_dim
        self.mixture_params = None
        self.per_traj_w = None
        self.per_step_w = None
        
        # Build MLP with final MDN layer
        self.mlp = ClassicMLP(context_dim=self.context_dim, hidden_dim=self.mlp_hidden_dim, out_dim=self.out_dim, activation=self.mlp_activation, dropout=self.mlp_dropout_prob)
        
        self.mdn_params = nn.Linear(self.out_dim, dims.mdn_emb_dim)
        self.per_traj_weights = nn.Linear(self.out_dim, cfg.num_modes)
        self.per_step_weights = nn.Linear(self.out_dim, cfg.num_modes * self.T)
        
        # Initialize MLP weights
        self._init_weights()
        
        
    def _init_weights(self):
        """
        Initialize linear layer weights with Xavier uniform and biases to zero.
        """
        
        return
    
    
    def forward(self, context, t=None, B=1):
        """
        Compute MDN parameters for input embeddings.
        Args:
            context (Tensor): Agent embeddings [B, input_embedding_dim].
        Returns:
            Tensor: MDN latent tensor [B, T, mixture_dim].
        """
        
        B = context.shape[0]
        shared_head = self.mlp(context)  # [B, mlp_out_dim]
        self.mixture_params = self.mdn_params(shared_head) # [B, T*mixture_dim]
        self.per_traj_w = self.per_traj_weights(shared_head)  # [B, M]
        self.per_step_w = self.per_step_weights(shared_head)  # [B, T*M]
        return 
    
    
    def sample(self, context=None, B=1):
        
        return self.mixture_params, self.per_traj_w, self.per_step_w
    
    
class MixtureTransformerDenoiser(Module):
    """
    Transformer-based denoiser producing mixture outputs.
    Args:
        cfg: Configuration object with attributes:
            - input_dim (int): Dim of each token.
            - d_model (int): Hidden dimension of transformer.
            - layer_size (int): Number of transformer layers.
            - nheads (int): Number of attention heads.
            - dropout (float): Dropout probability.
        T (int): Sequence length (forecast horizon).
    """
    
    def __init__(self, cfg, dims):
        
        super().__init__()
        
        self.input_dim = cfg.mixture_dim
        self.d_model = cfg.model.mdn_diffusion_d_model
        self.num_layers = cfg.model.mdn_diffusion_layer_size
        self.nheads = cfg.model.mdn_diffusion_nheads
        self.ff_dim = cfg.model.mdn_diffusion_ff_dim
        self.dropout = cfg.model.mdn_diffusion_dropout
        self.context_dim = dims.mdn_context_dim
        self.time_emb_dim = 3
        
        # Learnable residual scale 
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        
        # Context enriched intermediate input projection
        self.concat1 = ConcatSquash(
            cfg=Box({"out_dim": self.d_model, "dropout": 0.0}),
            dims=dims,
            query_dim= self.input_dim + self.time_emb_dim,
            key_dim=self.context_dim
        )
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=0.0,
            max_len=32,
            transpose=True
        )
        
        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nheads,
            dropout=self.dropout,
            dim_feedforward=self.ff_dim
        )
        
        # Transformer encoder model
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            num_layers=self.num_layers
            )
        
        # Context enriched intermediate input projection
        self.concat2 = ConcatSquash(
            cfg=Box({"out_dim": self.input_dim, "dropout": 0.0}),
            dims=dims,
            query_dim=self.d_model,
            key_dim=self.context_dim
        )
        
        # Initialize everything
        self.init_weights()
        
        
    def init_weights(self):
        """
        Xavier‐uniform for all Linear weights (zero biases),
        plus proper init for any MultiheadAttention inside the transformer.
        """
        
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            elif isinstance(m, nn.MultiheadAttention):
                
                # input q/k/v projections
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                # output projection
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0.0)
        
        
    def forward(self, x, beta, context=None):
        """
        Forward pass for mixture transformer denoiser.
        Args:
            x (Tensor): Noisy input, shape [B, T, input_dim].
            beta (Tensor): Noise parameter, shape [B, 1].
            context (None): Not implemented here.
        Returns:
            Tensor: Denoised output, shape [B, T, input_dim].
        """
        
        B, T, _ = x.shape
        
        # Time embedding: (β, sin(β), cos(β))
        t_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1).expand(B, T, -1)  # [B,T,time_emb_dim]
        
        # Unify context shape
        if context is not None:
            
            if context.dim() == 2:
                
                ctx_emb = context.view(B,1,-1).expand(B, T, -1)  # [B,T,context_dim]
                
            else:
                
                ctx_emb = context  # [B,T,context_dim]
        else:
            
            ctx_emb = torch.zeros(B, T, 0, device=x.device)  # [B,T,context_dim]
        
        # Save input for residual skip
        x_in = x
        
        # Project to model dim, permute to [sequence, batch, feature] for transformer, 
        # apply positional encoding, apply model and permute back to [batch, sequence, feature]
        x_proj, _ = self.concat1(queries=torch.cat([x_in, t_emb], dim=-1), keys=ctx_emb)  # [B,T,d_model]
        emb = self.pos_encoder(x_proj.permute(1, 0, 2))  # [T,B,d_model+time_emb_dim]
        trans = self.transformer_encoder(emb).permute(1, 0, 2)  # [B,T,d_model+time_emb_dim]
        
        # Project back to token dim
        x_pred, _ = self.concat2(queries=trans, keys=ctx_emb)  # [B,T,input_dim]
        
        # Global residual skip
        x0 = x_in + self.res_scale * x_pred
        return x0
    
    
class MixtureDiffusion(Module):
    """
    Mixture diffusion model.
    Args:
        cfg: Config with attributes:
            - steps (int): Number of diffusion steps.
            - beta_start (float): Starting beta value.
            - beta_end (float): Ending beta value.
            - mode (str): 'linear' or 'cosine' schedule.
        len_forecast_horizon (int): Sequence length T.
    """
    
    def __init__(self, cfg, dims):
        
        super().__init__()
        self.cfg = cfg
        self.num_steps = cfg.model.mdn_diffusion_steps
        self.beta_start = cfg.model.mdn_diffusion_beta_start
        self.beta_end = cfg.model.mdn_diffusion_beta_end
        self.mode = cfg.model.mdn_diffusion_mode
        self.context_dim = dims.mdn_context_dim
        self.importance_sampling = cfg.model.mdn_diffusion_importance_sampling
        self.importance_sampling_decay = cfg.model.mdn_diffusion_importance_sampling_decay
        self.dpm_solver_order = cfg.model.mdn_diffusion_dpm_solver_order
        self.mixture_dim = cfg.mixture_dim
        self.sampling_method = cfg.model.mdn_diffusion_sampling_method
        self.sampling_steps = cfg.model.mdn_diffusion_sampling_steps
        
        self.T = cfg.len_forecast_horizon
        self.device = cfg.target_device
        
        # Variance scheduler for noise schedule
        self.variance_scheduler = VarianceSchedule(
            num_steps=self.num_steps,
            mode=self.mode,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            importance_sampling = self.importance_sampling,
            decay=self.importance_sampling_decay
        )
        
        # Instantiate denoiser
        self.denoiser = MixtureTransformerDenoiser(cfg=cfg, dims=dims)
        self.mdn_params = nn.Linear(dims.mdn_emb_dim, dims.mdn_emb_dim)
        self.per_traj_weights = nn.Linear(dims.mdn_emb_dim, cfg.num_modes)
        self.per_step_weights = nn.Linear(dims.mdn_emb_dim, cfg.num_modes * self.T)
        
        # Initialize DPM-Solver scheduler for inference
        if self.sampling_method == 'dpm_ode':
            
            self.dpm_scheduler = DPMSolverMultistepScheduler(
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule="linear",
                num_train_timesteps=self.num_steps,
                solver_order=self.dpm_solver_order,
                prediction_type="epsilon",
                algorithm_type="dpmsolver++"
                )
            
        elif self.sampling_method == "dpm_sde":
            
            self.dpm_scheduler = EulerAncestralDiscreteScheduler(
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule="linear",
                num_train_timesteps=self.num_steps
            )
        
        
    def forward(self, context, t=None, B=1):
        """
        Training forward: add noise and predict it.
        Args:
            x0 (Tensor): Original input, shape [B,T,token_dim].
            context (Tensor, optional): Additional context, shape [B,context_dim].
            t (int or Tensor, optional): Timestep(s). If None, sampled uniformly.
            
        Returns:
            Tuple[Tensor,Tensor]: (e_rand, e_theta)
                e_rand: Random noise added, shape [B,T,token_dim].
                e_theta: Predicted noise, shape [B,T,token_dim].
        """
        
        #x0 = torch.randn(B, self.T, self.mixture_dim, device=self.device)  # [B,T,token_dim]
        x0 = torch.zeros(B, self.T, self.mixture_dim, device=self.device)  # [B,T,token_dim]
        
        # Sample or validate timesteps
        if t is None:
            
            # Choose between uniform or weighted sampling
            if self.variance_scheduler.importance_sampling:
                
                t_list = self.variance_scheduler.weighted_sample_t(B)
                
            else:
                
                t_list = self.variance_scheduler.uniform_sample_t(B)
                
            # Keep the list around so training loop can update EMAs
            self._last_t_list = t_list
            t = torch.tensor(t_list, device=self.device)
            
        else:
            
            t = t if torch.is_tensor(t) else torch.tensor(t, device=self.device)
        
        # Get alpha, beta for each sample
        alpha_bar = self.variance_scheduler.alpha_bars[t]  # [B]
        beta = self.variance_scheduler.betas[t].view(B,1,1)  # [B,1,1]
        
        # Coefficients for noisy input
        c0 = torch.sqrt(alpha_bar).view(B,1,1)  # [B,1,1]
        c1 = torch.sqrt(1 - alpha_bar).view(B,1,1)  # [B,1,1]
        
        # Sample and add noise
        e_rand = torch.randn_like(x0)  # [B,T,token_dim]
        x = c0 * x0 + c1 * e_rand  # [B,T,token_dim]
        
        # Predict noise using noise prediction model
        e_theta = self.denoiser(x=x, beta=beta, context=context)
        
        # Compute per-sample loss for this t
        if self.variance_scheduler.importance_sampling:
            
            # Mean over all other dims
            per_sample_loss = F.mse_loss(e_theta, e_rand, reduction="none").mean(dim=[1,2])  # [B]
            
            # update EMA buffer
            self.variance_scheduler.update_step_losses(self._last_t_list, per_sample_loss)
        
        return
    
    
    def sample(self, context, B):
        """
        Reverse diffusion sampling (DDPM or DDIM).
        Args:
            xT (Tensor): Initial noise, shape [B,T,token_dim].
            sampling (str): 'ddim' or 'ddpm'.
            sampling_steps (int): Number of sampling steps.
            context (Tensor, optional): Context, shape [B,context_dim].
        Returns:
            Tensor: Denoised output, shape [B,T,token_dim].
        """
        
        # Prepare initial sample batch
        #xT = torch.randn(B, self.T, self.mixture_dim, device=self.device)  # [B,T,token_dim]
        xT = torch.zeros(B, self.T, self.mixture_dim, device=self.device)  # [B,T,token_dim]
        
        # DPM-Solver branch
        if self.sampling_method == "dpm_ode" or self.sampling_method == "dpm_sde":
            
            # Configure inference timesteps
            self.dpm_scheduler.set_timesteps(num_inference_steps=self.sampling_steps)
            
            # Initial sample and iterate
            x = xT
            
            for t in self.dpm_scheduler.timesteps:
                
                t_int = int(t.item())
                beta = self.variance_scheduler.betas[t_int].view(1,1,1).expand(x.shape[0],1,1)
                eps = self.denoiser(x=x, beta=beta, context=context)
                self.dpm_scheduler.scale_model_input(sample=x, timestep=t)
                out = self.dpm_scheduler.step(model_output=eps,timestep=t, sample=x)
                x = out.prev_sample
                
                
            mixture_params = self.mdn_params(x.view(B, self.T * self.mixture_dim))  # [B, T*mixture_dim]
            per_traj_w = self.per_traj_weights(x.view(B, self.T * self.mixture_dim))  # [B,M]
            per_step_w = self.per_step_weights(x.view(B, self.T * self.mixture_dim))  # [B,T*M]
            return mixture_params, per_traj_w, per_step_w
        
        # DDPM, DDIM denoising branches
        else:
            
            stride = math.ceil(self.num_steps / self.sampling_steps)
            t_cur = self.num_steps
            x = xT
            
            # Denoising loop
            while t_cur > 0:
                
                # Next time step
                t_next = max(t_cur - stride, 0)
                
                # Scheduler parameters
                beta = self.variance_scheduler.betas[t_cur].view(1,1,1).expand(B,1,1)  # [B,1,1]
                alpha = self.variance_scheduler.alphas[t_cur]
                ab = self.variance_scheduler.alpha_bars[t_cur]
                ab_next = self.variance_scheduler.alpha_bars[t_next]
                
                # Predict noise
                eps_theta = self.denoiser(x=x, beta=beta, context=context)  # [B,T,token_dim]
                
                if self.sampling_method == "ddpm":
                    
                    sigma = self.variance_scheduler.get_sigmas(t_cur, flexibility=0.0).to(self.device)
                    z = torch.randn_like(x) if t_cur > 1 else torch.zeros_like(x)
                    c0 = 1.0 / math.sqrt(alpha)
                    c1 = (1 - alpha) / math.sqrt(1 - ab)
                    x = c0 * (x - c1 * eps_theta) + sigma * z  # [B,T,token_dim]
                    
                elif self.sampling_method == 'ddim':
                    
                    x0_pred = (x - torch.sqrt(1 - ab) * eps_theta) / torch.sqrt(ab)  # [B,T,token_dim]
                    x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_theta  # [B,T,token_dim]
                    
                # Set next time step
                t_cur = t_next
                
            mixture_params = self.mdn_params(x.view(B, self.T * self.mixture_dim))
            per_traj_w = self.per_traj_weights(x.view(B, self.T * self.mixture_dim))
            per_step_w = self.per_step_weights(x.view(B, self.T * self.mixture_dim))
            return mixture_params, per_traj_w, per_step_w