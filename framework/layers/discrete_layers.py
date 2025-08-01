import torch
import torch.nn as nn

from layers.shared_layers import SpecialMLP
from misc.utils import get_mixture_components


# ****************************************
# Overview:
#
# Hypotheses Generator Class: Section 3.6
#
# ****************************************


class HypothesesGenerator(nn.Module):
    """Generator that outputs K discrete trajectory hypotheses: the deterministic
    means of the *active* mixture modes followed by stochastic, offset-generated
    trajectories to make up the total using reparameterization trick.
    """
    
    def __init__(self, cfg, dims):
        
        super().__init__()
        
        # Params
        self.cfg     = cfg
        self.K       = cfg.num_hypos
        self.min_k   = cfg.min_k_per_mode
        self.M       = cfg.num_modes
        self.T       = cfg.len_forecast_horizon
        self.ctx_dim = dims.discrete_context_dim
        
        self.var_hidden   = cfg.model.discrete_generator_decoder_var_hidden_dim
        self.scale_hidden = cfg.model.discrete_generator_encoder_scale_hidden_dim
        self.act          = cfg.model.discrete_generator_mlp_activation
        self.dropout      = cfg.model.discrete_generator_mlp_dropout
        self.scale_out    = cfg.model.discrete_generator_scale_out_dim
        
        # Scale Encoder
        self.scale_encoder = SpecialMLP(
            context_dim=self.T,
            hidden_dim=self.scale_hidden,
            out_dim=self.scale_out,
            activation="ReLU",
            dropout=0.0
        )
        
        # Offset generator/Variance Decoder
        # maximum generated offsets per mode = K-1 (means occupy one slot)
        feat_dim = self.ctx_dim + self.scale_out
        out_dim  = (self.K - 1) * self.T * 2
        
        self.var_decoder = SpecialMLP(
            context_dim=feat_dim,
            hidden_dim=self.var_hidden,
            out_dim=out_dim,
            activation=self.act,
            dropout=self.dropout
        )
        
        # Init
        # Weight init
        self.apply(self._init_weights)
        return
        
        
    # Initializer
    def _init_weights(self, m):
        
        # General Linear init
        if isinstance(m, nn.Linear):
            if m in self.scale_encoder.modules():
                
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                
            elif m in self.var_decoder.modules():
                
                gain = nn.init.calculate_gain(self.act.lower())
                nn.init.xavier_uniform_(m.weight, gain=gain)
                
            else:
                
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                
            nn.init.constant_(m.bias, 0.0)
            
        # Small-offset biasing for the very last var_decoder layer
        if m is getattr(self.var_decoder, "layers", None) and hasattr(m, "__getitem__"):
            
            # no-op guard
            return
        
        final = self.var_decoder.layers[-1]
        
        if m is final:
            
            # push its outputs very close to zero
            nn.init.constant_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.0)
        
        return
    
    
    def forward(self, context_emb, per_anchor_mixture, prune_delta, top_p_trunc):
        
        # Get shapes
        B, M, T, K = context_emb.size(0), self.M, self.T, self.K
        device = context_emb.device
        min_extra = max(self.min_k - 1, 0)
        
        # Get MDN outputs
        loc, varf = get_mixture_components(per_anchor_mixture)
        mu    = loc.view(B, M, T, 2)  # [B,M,T,2]
        var   = varf.view(B, M, T, 2)
        sigma = torch.sqrt(var.mean(-1, keepdim=True))  # [B,M,T,1]
        mix_p = per_anchor_mixture.mixture_distribution.probs  # [B,M]
        
        active = mix_p >= prune_delta  # [B,M]
        no_active = (active.sum(dim=1) == 0)
        
        if no_active.any():
            
            fallback = mix_p.argmax(dim=1)
            active[no_active, fallback[no_active]] = True
            
        # Renormalize active probabilities
        p_active = mix_p * active.to(mix_p.dtype)
        p_active = p_active / p_active.sum(dim=1, keepdim=True)
        
        if top_p_trunc < 1.0:
            
            # Sort probs descending for each batch row
            sorted_p, sorted_idx = p_active.sort(dim=1, descending=True)
            
            # Cumulative mass
            cum_p = sorted_p.cumsum(dim=1)
            
            # Keep until mass exceeds threshold; always keep the first mode
            keep = cum_p <= top_p_trunc
            keep[:, 0] = True
            
            # Map back to original ordering
            mask = torch.zeros_like(p_active, dtype=torch.bool)
            mask.scatter_(1, sorted_idx, keep)
            
            # Zero-out the tails and renormalise
            p_active = torch.where(mask, p_active, torch.zeros_like(p_active))
            p_active = p_active / p_active.sum(dim=1, keepdim=True)
        
        # Per-row allocation
        Kb    = active.sum(dim=1)  # [B]
        K_gen = (K - Kb)  # [B]
        free  = (K_gen - min_extra * Kb).clamp(min=0)  # [B]
        alloc_int = torch.floor(p_active * free.unsqueeze(1))
        rem = (free - alloc_int.sum(dim=1)).long()  # [B]
        frac = p_active * free.unsqueeze(1) - alloc_int  # [B,M]
        _, order = frac.sort(dim=1, descending=True)
        rank = torch.arange(M, device=device).view(1, -1)
        take = rank < rem.unsqueeze(1)
        bonus = torch.zeros_like(alloc_int, dtype=torch.long)
        bonus.scatter_(1, order, take.long())
        k_gen = (min_extra + alloc_int.long() + bonus) * active.long()  # [B,M]
        
        # Decode offsets in batch
        ctx_rep = context_emb.unsqueeze(1).expand(-1, M, -1).reshape(B*M, -1)
        sigma_flat = sigma.view(B*M, T)
        h_sigma = self.scale_encoder(sigma_flat)
        feats = torch.cat([ctx_rep, h_sigma], dim=1)  # [B*M, feat_dim]
        
        offsets = self.var_decoder(feats)  # [B*M, (K-1)*T*2]
        offsets = offsets.view(B, M, K-1, T, 2)  # [B,M,K-1,T,2]
        
        # Reparameterize & mask offsets
        gen = mu.unsqueeze(2) + sigma.unsqueeze(2) * offsets  # [B,M,K-1,T,2]
        slots = torch.arange(K-1, device=device).view(1,1,-1)
        mask_off = slots < k_gen.unsqueeze(-1)  # [B,M,K-1]
        gen = gen * mask_off.unsqueeze(-1).unsqueeze(-1)
        
        # Stack means + generated
        means    = mu.unsqueeze(2)  # [B,M,1,T,2]
        all_hyps = torch.cat([means, gen], dim=2)  # [B,M,K,T,2]
        
        # Flatten & gather first K per batch
        B2, M2, L, T2, C = all_hyps.shape  # L == K
        flat = all_hyps.view(B2, M2*L, T2, C)  # [B,M*K,T,2]
        valid = torch.cat([active.unsqueeze(-1),  mask_off], dim=2).view(B2, M2*L)
        
        inf = torch.where(valid, torch.arange(M2*L, device=device), torch.full((B2, M2*L), M2*L, device=device))
        idxs = inf.argsort(dim=1)[:, :K]  # [B,K]
        
        hypos = flat.gather(1, idxs.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T2,C))  # [B,K,T,2]
        
        # Active means tensor
        active_mu = mu * active.unsqueeze(-1).unsqueeze(-1).to(mu.dtype)  # [B,M,T,2]
        
        # Return hypotheses, active means, and active-mode mask
        return hypos, active_mu, active