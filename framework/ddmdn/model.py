import torch

from torch.nn import Module
from misc import *
from layers import *
from box import Box


class DDMDN(Module):
    """
    DDMDN class for a dual Mixture Density Network (MDN) with diffusion-based backbone for trajectory forecast.
    """
    
    def __init__(self, cfg):
        """
        Initialize DDMDN.
        Args:
            cfg (Box): Config parameters.
        """
        
        super().__init__()
        self.cfg = cfg
        self.I = cfg.input_horizon
        self.T = cfg.len_forecast_horizon
        self.K = cfg.num_hypos
        self.mixture_dim = cfg.mixture_dim
        self.dims = self._set_shapes(cfg=cfg)
        
        # --- Input data handling
        # Temporal Encoder
        temporal_layer = getattr(encoding_layers, cfg.model.temporal_encoder_layer)
        self.TemporalEncoder = temporal_layer(cfg=cfg_by_prefix(cfg=cfg.model, prefix="temporal_encoder_"))
        
        # Spatial Encoder
        if self.cfg.with_spatial:
            
            spatial_layer = getattr(encoding_layers, cfg.model.cnn_encoder_layer)
            self.GridEncoder = spatial_layer(cfg=cfg_by_prefix(cfg=cfg.model, prefix="cnn_encoder_"))
        
        
        # --- Dual MDN Handling
        # Set MDN Distribution type, backbone and heads
        mdn_layer = getattr(mixture, cfg.model.mdn_mixture_typ)
        self.MDN = mdn_layer(cfg=cfg, dims=self.dims)
        
        
        # --- Discrete Hypotheses Handling
        if self.cfg.with_discrete:
            
            # Social Encoder
            if self.cfg.with_discrete_social_encoder:
                
                social_layer = getattr(social_layers, cfg.model.social_encoder_layer)
                self.SocialEncoder = social_layer(cfg=cfg_by_prefix(cfg=cfg.model, prefix="social_encoder_"))
            
            # Discrete Hypotheses Generator
            self.HypothesesGenerator = HypothesesGenerator(cfg=cfg, dims=self.dims)
        
        return
        
        
    def forward(self, agent_past, gt, agent_grid, agent_mask, prune_delta, prune_tau, top_p_trunc):
        """
        Forward pass for trajectory prediction.
        Args:
            agent_past (Tensor): Historical agent trajectories [B, A, I, 6].
            gt (Tensor): Ground truth future trajectories [B, A, T, 4].
            agent_grid (Tensor): Scene grid [B, 3, 128, 128].
            agent_mask (Tensor): Agent masking False=Valid, True=Invalid [B, A].
        Returns:
            dict: model outputs
        """
        
        # Get shapes
        B, _, _, _ = gt.shape
        
        # --- Input handling
        # Temporal Encoder
        temporal_emb, temporal_attn_weights = self.TemporalEncoder(x=agent_past)
        
        # Spatial Encoder and input feature fusion
        if self.cfg.with_spatial:
            
            spatial_emb, spatial_attn_weights, spatial_pool_weights = self.GridEncoder(x=agent_grid)
            mdn_context_emb = torch.cat([temporal_emb[:,0,:], spatial_emb], dim=1)
        
        # Without spatial
        else:
            
            mdn_context_emb = temporal_emb[:,0,:]
            spatial_attn_weights = None
            spatial_pool_weights = None
        
        
        # --- Dual MDN Handling
        # Get per-timestep and per-anchor-trajectory forecasts
        per_step_mixture, per_step_mixture_flat, per_anchor_mixture, per_step_mixture_detached, per_anchor_mixture_detached, mixture_success = self.MDN(
            context=mdn_context_emb, 
            prune_delta=prune_delta, 
            prune_tau=prune_tau
            )
        
        # --- Discrete Hypotheses Handling
        if self.cfg.with_discrete:
            
            # With Social Encoder
            if self.cfg.with_discrete_social_encoder:
                
                discrete_social_attn_emb, social_attn_weights = self.SocialEncoder(
                    x=temporal_emb.detach(),
                    m=agent_mask
                )
                
                discrete_context_emb = torch.cat([temporal_emb[:,0,:].detach(), discrete_social_attn_emb], dim=1)
            
            # Without Social Encoder
            else:
                
                discrete_context_emb = temporal_emb[:,0,:].detach()
                social_attn_weights = None
            
            # Discrete Hypotheses Generator
            d_hypo, d_top_mode_hypos, d_top_mask = self.HypothesesGenerator(
                context_emb=discrete_context_emb,
                per_anchor_mixture=per_anchor_mixture_detached,
                prune_delta=prune_delta,
                top_p_trunc=top_p_trunc,
                )
            
            ## --- Get Probabilities
            sorted_hypos, sorted_hypo_probs = self.score_hypotheses_by_confidence_levels(
                mixtures=per_step_mixture_detached, 
                hypos=d_hypo, 
                num_samples=self.cfg.train.num_samples, 
                bin_width=0.05
                )
            
        else:
            
            # Dummy placeholder
            sorted_hypos = None
            sorted_hypo_probs = None
            d_top_mode_hypos = None
            d_top_mask = None
            social_attn_weights = None
        
        # Result Storage
        res = {
            "mdn_success": mixture_success,  # boolean
            "mdn_per_step_mixture": per_step_mixture,  # [B, T, M, ...]
            "mdn_per_step_mixture_flat": per_step_mixture_flat,  # [B*T, M, ...]
            "mdn_per_step_mixture_detached": per_step_mixture_detached,  # [B, T, M, ...]
            "mdn_per_anchor_mixture": per_anchor_mixture,
            "hypos": sorted_hypos,  # [B, K, T, 2]
            "hypos_prob": sorted_hypo_probs, #  [B, K]
            "top_mode_hypos": d_top_mode_hypos,
            "top_mode_hypos_mask": d_top_mask,
            "temporal_attn_weights": temporal_attn_weights,
            "spatial_attn_weights": spatial_attn_weights,
            "spatial_pool_weights": spatial_pool_weights,
            "social_attn_weights": social_attn_weights,
        }
        
        return res, mixture_success
    
    
    def compute_loss(self, cfg, model_output, gt, wta_tau, prefix="Train"):
        """
        Compute the combined loss for MDN and diffusion components.
        Args:
            cfg (Box): Configuration object with len_forecast_horizon attribute.
            model_output (dict): Output from forward pass.
            gt (Tensor): Ground truth trajectories [B, A, T, 2].
        Returns:
            Loss scores: total_loss (Tensor), scores (dict), success (bool)
        """
        
        total_loss = torch.tensor(0., device=gt.device)
        scores = {f"{prefix}_Loss": 0.0}
        target_gt = gt[:, 0, :, :]  # [B, T, 2]
        
        # Get model outputs
        per_step_mixture = model_output['mdn_per_step_mixture']
        per_step_mixture_detached = model_output['mdn_per_step_mixture_detached']
        per_anchor_mixture = model_output['mdn_per_anchor_mixture']
        mdn_success = model_output['mdn_success']
        hypos = model_output['hypos']
        
        # Check mixture build
        if not mdn_success:
            return None, None, False
        
        # --- MDN Mixtures
        # NLL losses
        if cfg.train.lambda_per_step_nll != 0.0:
            
            per_step_nll = compute_per_step_nll(mixture=per_step_mixture, gt=target_gt)
            total_loss += cfg.train.lambda_per_step_nll * per_step_nll
            scores['mixture_per_step_NLL'] = per_step_nll.item()
        
        
        if cfg.train.lambda_per_anchor_nll != 0.0:
            
            per_traj_nll = compute_per_traj_nll(mixture=per_anchor_mixture, gt=target_gt)
            total_loss += cfg.train.lambda_per_anchor_nll * per_traj_nll
            scores['mixture_per_traj_NLL'] = per_traj_nll.item()*0.1
            
            
        # # --- Discrete Hypotheses Generation
        if self.cfg.with_discrete:
            
            # Discrete min MSE loss
            if cfg.train.lambda_d_minmse != 0.0:
                
                d_minmse_score = compute_discrete_initial_minMSE(cfg=self.cfg, hypos=hypos, gt=target_gt)
                total_loss += cfg.train.lambda_d_minmse * d_minmse_score
                scores['discrete_minMSE'] = d_minmse_score.item()
                
            # Discrete Inside‑set penalty
            if cfg.train.lambda_d_inside != 0.0:
                
                init_inside_loss = compute_inside_penalty(mixture=per_step_mixture_detached, hypos=hypos, conf_level=cfg.train.inside_quantile, num_samples=cfg.train.num_samples)
                total_loss += cfg.train.lambda_d_inside * init_inside_loss
                scores['discrete_Inside'] = init_inside_loss.item()
                
            # Discrete Annealed WTA
            if cfg.train.lambda_d_wta != 0.0:
                
                d_wta_score = compute_wta(cfg=cfg, hypos=hypos, gt=target_gt, wta_tau=wta_tau)
                total_loss += cfg.train.lambda_d_wta * d_wta_score
                scores['discrete_WTA'] = d_wta_score.item()
            
        # Total loss
        if isinstance(total_loss, torch.Tensor): scores[f"{prefix}_Loss"] = total_loss.item()
        else: scores[f"{prefix}_Loss"] = float(total_loss)
        
        return total_loss, scores, mdn_success
    
    
    def score_hypotheses_by_confidence_levels(self, mixtures, hypos, num_samples=1000, bin_width=0.05):
        """
        Estimate confidence for each hypothesis trajectory via MoG density sampling.
        Steps:
        1. Sample `num_samples` points from the mixture at each t.
        2. Compute densities for samples and hypotheses.
        3. Calculate percentile of each hypo-density among samples -> frac_ge [B,K,T].
        4. Bin percentiles of width `bin_width` and map to confidence = 1 - bin*bin_width.
        5. Average over time -> final_conf [B, K].
        6. Sort confidences descending.
        Args:
            mixtures:     MixtureSameFamily with batch_shape=[B, T], event_shape=[D].
            hypos (Tensor): Hypotheses tensor of shape [B, K, T, D].
            num_samples (int): Number of Monte Carlo samples per (b,t).
            bin_width (float): Width of each confidence bin (e.g. 0.05).
        Returns:
            sorted_hypos (Tensor): [B, K, T, D], hypotheses sorted by descending confidence.
            sorted_conf  (Tensor): [B, K], averaged confidences in [0,1].
        """
        
        B, K, T, D = hypos.shape
        # Sample from the mixture
        samples = mixtures.sample(sample_shape=(num_samples,))  # [N,B,T,D]
        
        # Compute densities for samples and hypos
        logp_samps = mixtures.log_prob(samples)  # [N,B,T]
        p_samps    = logp_samps.exp()  # [N,B,T]
        logp_hypos = mixtures.log_prob(hypos.permute(1,0,2,3))  # [K,B,T]
        p_hypos    = logp_hypos.exp().permute(1,0,2)  # [B,K,T]
        
        # Percentile: fraction of samples >= hypo-density
        p_samps_bt = p_samps.permute(1,2,0).unsqueeze(1)  # [B,1,T,N]
        p_h_bt     = p_hypos.unsqueeze(-1)  # [B,K,T,1]
        frac_ge    = (p_samps_bt >= p_h_bt).float().mean(dim=-1)  # [B,K,T]
        
        # Bin percentiles and map to confidence
        bin_idx   = torch.clamp((frac_ge / bin_width).ceil().long() - 1, min=0, max=int(1/bin_width)-1)  # [B,K,T]
        bin_upper = (bin_idx + 1).float() * bin_width  # [B,K,T]
        conf_bt   = 1.0 - bin_upper  # [B,K,T]
        
        # Average over time
        final_conf = conf_bt.mean(dim=-1)  # [B,K]
        
        # Sort descending
        sorted_conf, idx = final_conf.sort(dim=-1, descending=True)  # [B,K]
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, T, D)  # [B,K,T,D]
        sorted_hypos = torch.gather(hypos, dim=1, index=idx_exp)  # [B,K,T,D]
        
        return sorted_hypos, sorted_conf
    
    
    def _set_shapes(self, cfg):
        """
        Initialize and set dimensional parameters for model components based on configuration.
        Computes temporal, spatial, social, MDN, and discrete context dimensions to populate a parameter box.
        Args:
            cfg (Config): Configuration object containing model architecture settings and dimension specifications.
        Returns:
            Box: A container with keys:
                - T (int): Forecast horizon length.
                - temporal_emb_dim (int): Dimension of temporal embeddings.
                - spatial_emb_dim (int): Dimension of spatial (CNN) embeddings.
                - social_emb_dim (int): Dimension of social encoder embeddings.
                - mdn_context_dim (int): MDN context input dimension.
                - mdn_mixture_dim (int): MDN mixture output dimension.
                - mdn_emb_dim (int): MDN embedding dimension.
                - discrete_context_dim (int): Discrete decoder context input dimension.
        """
        
        params = {}
        params["T"] = cfg.len_forecast_horizon
        
        # --- Inputs
        # Temporal embedding dim
        if cfg.model.temporal_encoder_bidirectional: params["temporal_emb_dim"] = cfg.model.temporal_encoder_out_dim * 2
        else: params["temporal_emb_dim"] = cfg.model.temporal_encoder_out_dim
        
        # Spatial embedding dim
        if self.cfg.with_spatial: params["spatial_emb_dim"] = cfg.model.cnn_encoder_out_dim
        else: params["spatial_emb_dim"] = 0
        
        # --- MDN
        # MDN Input Context dim
        params["mdn_context_dim"] = params["temporal_emb_dim"] + params["spatial_emb_dim"]
        
        # MDN Output embedding dim
        params["mdn_mixture_dim"] = cfg.mixture_dim
        params["mdn_emb_dim"] = cfg.mixture_dim * self.T
        
        
        # --- Discrete
        # Social embedding dim
        if self.cfg.with_discrete_social_encoder: params["social_emb_dim"] = cfg.model.social_encoder_out_dim + params["temporal_emb_dim"] 
        else: params["social_emb_dim"] = params["temporal_emb_dim"] 
        
        # Discrete Input Context dim
        params["discrete_context_dim"] = params["social_emb_dim"]
        
        # Discrete Refiner Context dim
        params['discrete_diffusion_context_dim'] = params["social_emb_dim"]
        
        return Box(params)
    