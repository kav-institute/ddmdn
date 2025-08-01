import torch
import torch.nn as nn
import layers.mdn_layers as mdn_layers

from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical


def differentiable_prune_logits(logits, delta=0.02, tau=0.1, eps=1e-8):
    """
    logits: [..., M]
    δ      : prune-threshold on the *post*-softmax probs
    τ      : temperature for the sigmoid gate
    returns: new_logits of same shape
    """
    
    # compute current probs
    p = torch.softmax(logits, dim=-1)
    # build a *soft* mask
    gate = torch.sigmoid((p - delta) / tau)
    # apply gate & renormalize
    p_tilde = p * gate
    p_new   = p_tilde / (p_tilde.sum(-1, keepdim=True) + eps)
    # go back to logit space
    return torch.log(p_new + eps)


class MDN_MultivariateNormal(nn.Module):
    """
    Mixture Density Network with full covariance Gaussian components.
    Args:
        cfg (Box): Configuration with attribute in_dim.
        len_forecast_horizon (int): Forecast horizon length T.
        num_modes (int): Number of mixture modes M.
    """
    
    def __init__(self, cfg, dims):
        """
        Initialize the MDN_MultivariateNormal layer.
        """
        
        super(MDN_MultivariateNormal, self).__init__()
        
        # Params
        self.cfg = cfg
        self.T = cfg.len_forecast_horizon
        self.M = cfg.num_modes
        self.variance_epsilon = cfg.model.mdn_variance_epsilon
        
        mdn_layer = getattr(mdn_layers, cfg.model.mdn_mixture_layer)
        self.MDN_Model = mdn_layer(cfg=cfg, dims=dims)
    
    
    def forward(self, context, prune_delta, prune_tau):
        """
        Compute MDN parameters for multivariate normal components.
        Args:
            context (Tensor): Context embeddings [B, context_dim].
        Returns:
            Tensor: MDN latent tensor [B, T, mixture_dim].
        """
        
        # Apply model
        B = context.shape[0]
        self.MDN_Model(context=context, B=B)
        
        # Get latent vector
        mixture_latent, per_traj_weight, per_step_weight = self.MDN_Model.sample(context=context, B=B)
        
        # Build distribution from latent
        per_step_mixture, per_step_mixture_flat, per_anchor_mixture, success = self._build_mixture(mixture_latent=mixture_latent, per_traj_weight=per_traj_weight, per_step_weight=per_step_weight, prune_delta=prune_delta, prune_tau=prune_tau)
        per_step_mixture_detached, _, per_anchor_mixture_detached, _ = self._build_mixture(mixture_latent=mixture_latent.detach().clone(), per_traj_weight=per_traj_weight.detach().clone(), per_step_weight=per_step_weight.detach().clone(), prune_delta=prune_delta, prune_tau=prune_tau)
        return per_step_mixture, per_step_mixture_flat, per_anchor_mixture, per_step_mixture_detached, per_anchor_mixture_detached, success 
    
    
    def _build_mixture(self, mixture_latent, per_traj_weight, per_step_weight, prune_delta, prune_tau):
        """
        Builds both a per-trajectory and a per-step MDN from raw outputs.
        Args:
            mixture_latent: Tensor [B, T, M*5]
            per_traj_weight: Tensor [B, M]
            per_step_weight: Tensor [B, T*M]
        Returns:
            per_anchor_mixture: MixtureSameFamily over 2T-dim Gaussians
            per_step_mixture: MixtureSameFamily over 2D Gaussians per timestep
            per_step_mixture_flat: flattened MixtureSameFamily over 2D Gaussians [B*T]
            success: bool
        """
        
        # Consistency check
        if torch.isnan(mixture_latent).any() or torch.isnan(per_traj_weight).any() or torch.isnan(per_step_weight).any():
            return None, None, None, False
        
        B = mixture_latent.size(0)
        
        # Reshape to [B, M, T, 5]
        p = mixture_latent.view(B, self.T, self.M, 5).permute(0, 2, 1, 3)
        mu_x = p[..., 0]  # [B, M, T]
        mu_y = p[..., 1]  # [B, M, T]
        log_sigma_x2 = p[..., 2]  # [B, M, T]
        log_sigma_y2 = p[..., 3]  # [B, M, T]
        rho = torch.tanh(p[..., 4])  # [B, M, T]
        
        # Variances
        var_x = log_sigma_x2.exp() + self.variance_epsilon  # [B, M, T]
        var_y = log_sigma_y2.exp() + self.variance_epsilon  # [B, M, T]
        sx = torch.sqrt(var_x)
        sy = torch.sqrt(var_y)
        
        # --- Per-trajectory mixture
        # Means
        loc = torch.stack([mu_x, mu_y], dim=-1).reshape(B, self.M, 2*self.T)  # [B, M, 2T]
        
        # Build block-diagonal covariance
        cov_full = torch.zeros(B, self.M, 2*self.T, 2*self.T, device=mixture_latent.device)  # [B, M, 2T, 2T]
        
        # Fill
        for t in range(self.T):
            
            i = 2*t; j = i+1
            off = rho[..., t] * sx[..., t] * sy[..., t]
            
            cov_full[:, :, i, i] = var_x[..., t]
            cov_full[:, :, j, j] = var_y[..., t]
            cov_full[:, :, i, j] = off
            cov_full[:, :, j, i] = off
        
        # With or without dynamic mode usage
        if self.cfg.with_dynamic_modes:
            
            # Apply differentiable pruning to trajectory logits
            traj_logits_gated = differentiable_prune_logits(per_traj_weight, delta=prune_delta, tau=prune_tau)  # [B, M]
            probs_traj = torch.softmax(traj_logits_gated, dim=-1)
            
        else:
            
            probs_traj = torch.softmax(per_traj_weight, dim=-1)  # [B, M]
        
        # Build mixture
        weights_traj = Categorical(probs=probs_traj)
        comp_traj = MultivariateNormal(loc=loc, covariance_matrix=cov_full)
        per_anchor_mixture = MixtureSameFamily(mixture_distribution=weights_traj, component_distribution=comp_traj)
        
        
        # --- Per-step mixture (batched over T)
        # Means
        mu = torch.stack([mu_x, mu_y], dim=-1).permute(0, 2, 1, 3)  # [B, T, M, 2]
        
        # Variances
        covs = torch.zeros(B, self.T, self.M, 2, 2, device=mixture_latent.device)
        
        # Fill
        for t in range(self.T):
            
            off_t = rho[..., t] * sx[..., t] * sy[..., t]
            
            covs[:, t, :, 0, 0] = var_x[..., t]
            covs[:, t, :, 1, 1] = var_y[..., t]
            covs[:, t, :, 0, 1] = off_t
            covs[:, t, :, 1, 0] = off_t
            
            
        # With or without dynamic mode usage
        if self.cfg.with_dynamic_modes:
            
            # Apply differentiable pruning to step logits
            step_logits = per_step_weight.view(B, self.T, self.M)
            flat_logits = step_logits.reshape(-1, self.M)  # [B*T, M]
            flat_gated = differentiable_prune_logits(flat_logits, delta=prune_delta, tau=prune_tau)  # [B*T, M]
            step_logits_gated = flat_gated.view(B, self.T, self.M)
            step_probs = torch.softmax(step_logits_gated, dim=-1)
            
        else:
            
            step_logits = per_step_weight.view(B,self.T,self.M)  # [B, T, M]
            step_probs = torch.softmax(step_logits, dim=-1)  # [B, T, M]
        
        # Non-flat per-step MixtureSameFamily
        weights_step = Categorical(probs=step_probs)
        comp_step = MultivariateNormal(loc=mu, covariance_matrix=covs)
        per_step_mixture = MixtureSameFamily(mixture_distribution=weights_step, component_distribution=comp_step)
        
        # --- Flat per-step mixture
        BT = B * self.T
        flat_mu = mu.reshape(BT, self.M, 2)
        flat_covs = covs.reshape(BT, self.M, 2, 2)
        flat_probs = step_probs.reshape(BT, self.M)
        weights_flat = Categorical(probs=flat_probs)
        comp_flat = MultivariateNormal(loc=flat_mu, covariance_matrix=flat_covs)
        per_step_mixture_flat = MixtureSameFamily(mixture_distribution=weights_flat, component_distribution=comp_flat)
        
        return per_step_mixture, per_step_mixture_flat, per_anchor_mixture, True
