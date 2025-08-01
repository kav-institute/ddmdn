import torch
import torch.nn as nn

from torch.nn import Module

# **************************
# Overview:
#
# Shared basic components
#
# **************************


class ClassicMLP(Module):
    """
    MLP backbone.
    Args:
        cfg: Configuration parameters.
    """
    
    def __init__(self, context_dim, hidden_dim, out_dim, activation, dropout):
        """
        Build sequential MLP based on user defined number of layers.
        Args:
            cfg: Configuration as above.
        """
        
        super(ClassicMLP, self).__init__()
        
        self.activation = getattr(nn, activation)()
        self.in_dim = context_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_prob = dropout
        
        # Define MLP setup
        dims = (self.in_dim,) + tuple(self.hidden_dim) + (self.out_dim,)
        self.layers = nn.ModuleList()
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Build MLP
        for i in range(len(dims)-1):
            
            if i < len(dims)-2:
                
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
                self.layers.append(self.activation)
                self.layers.append(self.dropout)
                
            else:
            
                # Last layer
                self.layers.append(nn.Linear(dims[i], self.out_dim))
        
        # Full model
        self.net = nn.Sequential(*self.layers)
        
        # Initialize MLP weights
        self.init_weights()
        
        
    def init_weights(self):
        """
        Xavier‐uniform initialization for linear layers in MLP,
        using the correct gain for the chosen activation.
        """
        
        # figure out the activation name, e.g. 'relu', 'tanh', etc.
        act_name = self.activation.__class__.__name__.lower()
        
        # compute gain automatically
        gain = nn.init.calculate_gain(act_name)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
                    
    def forward(self, x):
        """
        Forward pass per agent.
        Args:
            x (Tensor): Input embeddings, shape [B, in_dim].
        Returns:
            Tensor: Output params, shape [B, out_dim].
        """
        
        out = self.net(x)
        return out  # [B, out_dim]
    
    
class SpecialMLP(Module):
    """
    MLP backbone.
    Args:
        cfg: Configuration parameters.
    """
    
    def __init__(self, context_dim, hidden_dim, out_dim, activation, dropout):
        """
        Build sequential special MLP based on user defined number of layers.
        Args:
            cfg: Configuration as above.
        """
        
        super(SpecialMLP, self).__init__()
        
        self.activation = getattr(nn, activation)()
        self.in_dim = context_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_prob = dropout
        
        # Define MLP setup
        dims = (self.in_dim,) + tuple(self.hidden_dim)  + (self.out_dim,)
        self.layers = nn.ModuleList()
        
        # Build MLP
        for i in range(len(dims) - 1):
            
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Initialize MLP weights
        self.init_weights()
        
        
    def init_weights(self):
        """
        Xavier‐uniform initialization for linear layers in MLP,
        using the correct gain for the chosen activation.
        """
        
        # figure out the activation name, e.g. 'relu', 'tanh', etc.
        act_name = self.activation.__class__.__name__.lower()
        
        # compute gain automatically
        gain = nn.init.calculate_gain(act_name)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
                    
    def forward(self, x):
        """
        Forward pass per agent.
        Args:
            x (Tensor): Input embeddings, shape [B, in_dim].
        Returns:
            Tensor: Output params, shape [B, out_dim].
        """
        
        for i in range(len(self.layers)):
            
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
            
        return x  # [B, out_dim]
    
    
class ConcatSquash(Module):
    """
    Gated linear layer with contextual hyper-networks.
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        cond_dim (int): Conditioning vector dimension.
    """
    
    def __init__(self, cfg, dims, query_dim, key_dim):
        """
        Build base linear and hyper-networks for bias and gate.
        Args:
            in_dim, out_dim, cond_dim: see class doc.
        """
        
        super(ConcatSquash, self).__init__()
        
        self.in_dim = query_dim
        self.out_dim = cfg.out_dim
        self.cond_dim = key_dim
        self.dropout_prob = cfg.dropout
        
        self._layer = nn.Linear(self.in_dim, self.out_dim)
        self._hyper_bias = nn.Linear(self.cond_dim, self.out_dim, bias=False)
        self._hyper_gate = nn.Linear(self.cond_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        
        # Base linear layer
        nn.init.xavier_uniform_(self._layer.weight)
        nn.init.constant_(self._layer.bias, 0.0)
        
        # Hyper-bias (no bias term on this one)
        nn.init.xavier_uniform_(self._hyper_bias.weight)
        
        # Hyper-gate
        nn.init.xavier_uniform_(self._hyper_gate.weight)
        nn.init.constant_(self._hyper_gate.bias, 0.0)
        
        
    def forward(self, queries, keys):
        """
        Apply gated linear transformation.
        Args:
            keys (Tensor): Conditioning tensor, shape [B, cond_dim].
            queries (Tensor): Input tensor, shape [B, in_dim].
        Returns:
            Tensor: Output tensor, shape [B, out_dim].
        """
        
        gate = torch.sigmoid(self._hyper_gate(keys))  # [B, out_dim]
        bias = self._hyper_bias(keys)  # [B, out_dim]
        ret = self._layer(queries) * gate + bias  # [B, out_dim]
        ret = self.dropout(ret)
        return ret, None
