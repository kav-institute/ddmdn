import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn import Module

# ************************************
# Overview:
#
# Input encoding: Section 3.3
#
# ************************************


class ConvAttnLSTM_Encoder(nn.Module):
    """
    Past trajectory encoder. Encodes agent's past trajectory into a fixed vector embedding.
    """
    def __init__(self, cfg):
        
        super().__init__()
        
        # Configuration
        self.in_dim = cfg.in_dim
        self.num_layers = cfg.num_layers
        self.bidirectional = cfg.bidirectional
        self.hidden_dim = cfg.out_dim
        self.conv_out_dim = cfg.conv_out_dim
        self.conv_kernel = cfg.conv_kernel
        self.conv_stride = cfg.conv_stride
        self.conv_padding = cfg.conv_padding
        self.attn_num_layer = cfg.attn_layer_size
        self.attn_nheads = cfg.attn_nheads
        self.with_spatial = cfg.with_spatial
        self.dropout_prob = cfg.dropout
        
        # Effective hidden dim after LSTM (accounts for bidirectionality)
        self.eff_hidden_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        # Spatial convolution
        if self.with_spatial:
            
            self.lstm_in_dim = self.conv_out_dim
            self.activation = nn.ReLU()
            
            self.conv = nn.Conv1d(
                in_channels=self.in_dim,
                out_channels=self.conv_out_dim,
                kernel_size=self.conv_kernel,
                stride=self.conv_stride,
                padding=self.conv_padding
            )
            
        else:
            
            self.activation = nn.Identity()
            self.conv = nn.Identity()
            self.lstm_in_dim = self.in_dim
            
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Self-attention layers (stacked)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.eff_hidden_dim,
                num_heads=self.attn_nheads,
                batch_first=True,
                dropout=0.0
            ) for _ in range(self.attn_num_layer)
        ])

            
        # Fusion layer: concatenates pooled attention + last hidden
        self.fusion = nn.Linear(2 * self.eff_hidden_dim,self.eff_hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        # Initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                gate_gain = nn.init.calculate_gain('sigmoid')
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=gate_gain)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(0.)
                        hidden_size = self.hidden_dim
                        param.data[hidden_size:2*hidden_size].fill_(1.)
                    elif 'bias_hh' in name:
                        nn.init.constant_(param.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
                    
    def forward(self, x):
        """
        Args:
            x (Tensor): Past trajectories, shape [B, A, I, 6].
        Returns:
            Tensor: Agent embeddings, shape [B, A, out_dim].
        """
        
        # Shape check
        is_4d = (x.dim() == 4)
        
        if is_4d:
            
            B, A, I, D = x.shape
            x_flat = x.view(B * A, I, D)
            
        else:
            
            B, A = x.shape[0], 1
            x_flat = x
            
        # Convolution + activation
        x_t = x_flat.transpose(1, 2)  # (B*A, D, I)
        conv_out = self.activation(self.conv(x_t))  # (B*A, conv_out_dim, L)
        lstm_input = conv_out.transpose(1, 2)       # (B*A, L, conv_out_dim)
        
        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(lstm_input)  # (B*A, L, eff_hidden_dim)
        
        # Self-attention stack
        attn_input = lstm_out  # (B*A, L, eff_hidden_dim)
        attn_weights = None
        
        for layer in self.self_attn_layers:
            
            attn_input, attn_weights = layer(attn_input, attn_input, attn_input)
            
        # Attentive pooling over time
        # Compute importance of each time step by averaging attention weights over queries
        importance = attn_weights.mean(dim=1)  # (B*A, L)
        alpha = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        pooled = torch.bmm(alpha.unsqueeze(1), attn_input).squeeze(1)  # (B*A, eff_hidden_dim)
        
        # Last LSTM hidden state
        if self.bidirectional:
            
            # Combine forward + backward for last layer
            h_forward = h_n[-2]  # (B*A, hidden_dim)
            h_backward = h_n[-1]  # (B*A, hidden_dim)
            last_hidden = torch.cat([h_forward, h_backward], dim=1)  # (B*A, eff_hidden_dim)
            
        else:
            
            last_hidden = h_n[-1]  # (B*A, eff_hidden_dim)
            
        # Fuse pooled attention features with last hidden state
        concat = torch.cat([pooled, last_hidden], dim=1)  # (B*A, 2*eff_hidden_dim)
        fused = self.fusion(concat)  # (B*A, eff_hidden_dim)
        out_flat = self.dropout(fused)  # (B*A, eff_hidden_dim)
        
        # Reshape back
        out = out_flat.view(B, A, self.eff_hidden_dim)  # [B, A, eff_hidden_dim]
        return out, attn_weights
    
     
class CNNEncoder(Module):
    """
    Occupancy grid encoder using an CNN [B, 3, H, W] => [B, N, out_dim] or [B, out_dim].
    """
    
    def __init__(self, cfg):
        
        super(CNNEncoder, self).__init__()
        
        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim
        self.nheads = cfg.nheads
        self.num_layers = cfg.layer_size
        self.dropout_prob = cfg.dropout
        
        self.cnn_encoder = nn.Sequential(
            
            nn.Conv2d(self.in_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
        )
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.out_dim, num_heads=self.nheads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
        # One LayerNorm per attention layer (for the residual + norm)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.out_dim) for _ in range(self.num_layers)
        ])
        
        # A learnable query vector for attention pooling
        self.global_query = nn.Parameter(torch.randn(self.out_dim))
        self.dropout = nn.Dropout(self.dropout_prob)
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.transpose(1, 2)
        
        # Initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        
        for m in self.modules():
            
            # 1) Conv2d: Kaiming‐uniform appropriate for ReLU
            if isinstance(m, nn.Conv2d):
                
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            # 2) BatchNorm2d: unit weight, zero bias
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
                
            # 3) MultiheadAttention: Xavier for all projections, zero biases
            elif isinstance(m, nn.MultiheadAttention):
                
                # in_proj (q,k,v) weights and bias
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    
                # out projection
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0.0)
        
        
    def forward(self, x):
        """
        x: [B, 3, H, W]
        return => [B, N, out_dim] or [B, out_dim]
        """
        
        # Spatial Encoder
        cnn_features = self.cnn_encoder(x)  # [B, out_dim, H, W]
        cnn_features = self.flatten(cnn_features)  # [B, out_dim, N]
        cnn_features = self.transpose(cnn_features)  # [B, N, out_dim]
        
        # Spatial Attention
        all_attn_weights = []
        out = cnn_features
        for i in range(self.num_layers):
            attn_layer = self.attention_layers[i]
            norm_layer = self.layer_norms[i]
            
            # MultiheadAttention expects (query, key, value). We do self‐attention:
            attn_out, attn_weights = attn_layer(out, out, out)  # [B, N, out_dim], [B, N, N]
            
            # Residual + LayerNorm
            out = norm_layer(attn_out + out)
            all_attn_weights.append(attn_weights)
        
        B, N, E = out.shape
        
        # Expand global_query for each batch instance: [B, E]
        query = self.global_query.unsqueeze(0).expand(B, E)
        
        # Compute dot product between global query and each spatial location in out: [B, N]
        scores = (out * query.unsqueeze(1)).sum(dim=-1)
        pool_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum to compute context vector: [B, E]
        context = torch.bmm(pool_weights.unsqueeze(1), out).squeeze(1)
        context = self.dropout(context)
        return context, attn_weights, pool_weights