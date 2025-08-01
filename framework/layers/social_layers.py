import torch
import math
import torch.nn as nn


# ******************************
# Overview:
#
# Social encoding - Section 3.3
#
# *****************************


class TargetOutWayAttention(nn.Module):
    """
    Cross-attention module with a null-key mechanism for a single target query over agent context.
    Adds an extra null column to ignore irrelevant context keys, then attends and fuses values back into the target.
    Args:
        d_model (int): Model embedding dimension.
        nhead (int): Number of attention heads.
        dropout (float): Dropout probability for attention weights.
    Forward Args:
        target (Tensor): Target query tensor of shape [B, D].
        context (Tensor): Context key/value tensor of shape [B, A, D].
        key_padding_mask (Tensor, optional): Mask tensor [B, A] where True indicates padding keys.
    Returns:
        enriched (Tensor): Updated target features of shape [B, D].
        attn (Tensor): Attention weights over real keys of shape [B, H, A].
    """
    
    def __init__(self, d_model, nhead, dropout):
        
        super().__init__()
        
        self.nhead = nhead
        self.d_model = d_model
        self.d_head = d_model // nhead
        
        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
        
        
    def init_weights(self):
        """
        Initialize all submodules:
        - Linear layers: Xavier‐uniform + zero biases
        - LayerNorm: weight=1, bias=0
        """
        
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        
    def forward(self, target, context, key_padding_mask=None):
        
        # Shapes
        B, A, D = context.size(0), context.size(1), self.d_model
        H, d = self.nhead, self.d_head
        
        # Project Q/K/V  
        Q = self.W_q(target).view(B, H, 1, d)  # [B,H,1,d]
        KV = self.W_kv(context).view(B, A, 2, H, d)  # [B,A,2,H,d]
        K, V = KV.unbind(dim=2)  # each [B,A,H,d]
        K = K.permute(0,2,1,3)  # [B,H,A,d]
        V = V.permute(0,2,1,3)  # [B,H,A,d]
        
        # Scaled dot-product  
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d) # [B,H,1,A]
        
        if key_padding_mask is not None:
            
            mask = key_padding_mask.view(B,1,1,A)  # [B,1,1,A]
            scores = scores.masked_fill(mask, float('-inf'))
            
        # Null column  
        null_col = torch.zeros_like(scores[..., :1])  # [B,H,1,1]
        scores = torch.cat([scores, null_col], dim=-1)  # [B,H,1,A+1]
        
        # Softmax & drop null  
        all_attn = torch.softmax(scores, dim=-1)  # [B,H,1,A+1]
        attn = all_attn[..., :-1]  # [B,H,1,A]
        attn = self.dropout(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)  # [B,H,1,A]
        
        # Weighted sum of values  
        out = attn @ V  # [B,H,1,d]
        out = out.reshape(B, H, d).transpose(1,2).reshape(B, D)  # [B,D]
        enriched = self.out(out)  # [B,D]
        
        return enriched, attn.squeeze(2)  # enriched:[B,D], attn:[B,H,A]
    
    
class OutWayCrossLayer(nn.Module):
    """
    Single cross-attention plus feed-forward network (FFN) layer with the OutWay null mechanism.
    Args:
        d_model (int): Model embedding dimension.
        nhead (int): Number of attention heads.
        d_ff (int): Inner dimension of the FFN.
        dropout (float): Dropout probability.
    Forward Args:
        target (Tensor): Target query tensor [B, D].
        context (Tensor): Context tensor [B, A, D].
        mask (Tensor, optional): Padding mask [B, A].
    Returns:
        output (Tensor): Updated target features [B, D].
        attn (Tensor): Attention weights [B, H, A].
    """
    
    def __init__(self, d_model, nhead, d_ff, dropout):
        
        super().__init__()
        
        self.attn = TargetOutWayAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.init_weights()
        
        
    def init_weights(self):
        """
        Initialize all submodules:
        - Linear layers: Xavier‐uniform + zero biases
        - LayerNorm: weight=1, bias=0
        """
        
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        
    def forward(self, target, context, mask=None):
        
        # Apply attention
        enriched, attn = self.attn(target, context, key_padding_mask=mask)
        
        h1 = self.norm1(target + enriched)  # [B,D]
        h2 = self.norm2(h1 + self.ff(h1))  # [B,D]
        return h2, attn  # [B,D], [B,H,A]
    
    
class SocialOutWayEncoder(nn.Module):
    """
    Encoder that stacks multiple OutWayCrossLayer layers to produce a socially enriched target embedding and attention maps.
    Args:
        cfg (Config): Configuration with the following attributes:
            - in_dim (int): Input feature dimension.
            - model_dim (int): Hidden/model embedding dimension.
            - out_dim (int): Output feature dimension.
            - head_size (int): Number of attention heads.
            - ff_dim (int): Feed-forward inner dimension.
            - layer_size (int): Number of cross-attention layers.
            - dropout (float): Dropout probability.
    Forward Args:
        x (Tensor): Full agent features of shape [B, A, in_dim] (target at index 0).
        m (Tensor): Mask tensor of shape [B, A], where True indicates padding.
    Returns:
        h_out (Tensor): Enriched target features of shape [B, out_dim].
        attn_stack (Tensor): Stacked attention maps of shape [L, B, H, A].
    """
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.in_dim = cfg.in_dim
        self.model_dim = cfg.model_dim
        self.out_dim = cfg.out_dim
        self.heads = cfg.head_size
        self.ff_dim = cfg.ff_dim
        self.num_layers = cfg.layer_size
        self.dropout = cfg.dropout
        
        # Input projection if in_dim != model_dim
        self.in_proj = (nn.Identity() if self.in_dim == self.model_dim else nn.Linear(self.in_dim, self.model_dim))
        
        # Output projection if out_dim != model_dim
        self.out_proj = (nn.Identity() if self.out_dim == self.model_dim else nn.Linear(self.model_dim, self.out_dim))
        
        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            OutWayCrossLayer(self.model_dim, self.heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        self.init_weights()
        
        
    def init_weights(self):
        """
        Initialize all submodules:
        - Linear layers: Xavier‐uniform + zero biases
        - LayerNorm: weight=1, bias=0
        """
        
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        
    def forward(self, x, m):
        
        h = self.in_proj(x)  # [B, A, D]
        target = h[:, 0, :]  # [B, D]
        context = h  # [B, A, D]
        
        attns = []
        for layer in self.layers:
            
            target, attn = layer(target, context, mask=m)
            attns.append(attn)
            
        h_out = self.out_proj(target)  # [B, out_dim]
        attn_stack = torch.stack(attns, dim=0)  # [L, B, H, A]
        return h_out, attn_stack