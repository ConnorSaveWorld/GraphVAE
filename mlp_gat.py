#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     mlp_gat.py
# Author:        Yang Run
# Created Time:  2022-12-06  13:48
# Last Modified: <none>-<none>
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GINConv
from dgl.nn import GATConv, GATv2Conv
from dgl.nn import EGATConv
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from utils import create_activation, create_norm


def exists(x):
    return x is not None


class FastDenoising_Unet(nn.Module):
    """Lightweight UNet using simpler operations for faster inference."""
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers=2,  # Fewer layers
                 activation='gelu',
                 feat_drop=0.1,
                 norm='layernorm',
                 ):
        super(FastDenoising_Unet, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = create_activation(activation)
        
        # 1. Initial Projection
        self.initial_proj = nn.Linear(in_dim + num_hidden, num_hidden)
        
        # 2. Use simpler MLP layers instead of GAT
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Downsampling: simple MLP with residual connection
            self.down_layers.append(nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.LayerNorm(num_hidden) if norm == 'layernorm' else nn.Identity(),
                self.activation,
                nn.Dropout(feat_drop),
                nn.Linear(num_hidden, num_hidden)
            ))
            
            # Upsampling: account for skip connections
            self.up_layers.append(nn.Sequential(
                nn.Linear(num_hidden * 2, num_hidden),  # *2 for skip connection
                nn.LayerNorm(num_hidden) if norm == 'layernorm' else nn.Identity(),
                self.activation,
                nn.Dropout(feat_drop),
                nn.Linear(num_hidden, num_hidden)
            ))
        
        # 3. Bottleneck
        self.middle_layer = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.LayerNorm(num_hidden) if norm == 'layernorm' else nn.Identity(),
            self.activation,
            nn.Linear(num_hidden, num_hidden)
        )
        
        # 4. Final projection
        self.final_proj = nn.Linear(num_hidden, out_dim)
        
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g, x_t, time_embed):
        # 1. Initial projection
        h = torch.cat([x_t, time_embed], dim=1)
        h = self.initial_proj(h)
        h = self.activation(h)
        
        # 2. Store skip connections
        skip_connections = []
        
        # 3. Downsampling path
        for layer in self.down_layers:
            skip_connections.append(h)
            residual = h
            h = layer(h)
            h = h + residual  # Residual connection
        
        # 4. Bottleneck
        residual = h
        h = self.middle_layer(h)
        h = h + residual
        
        # 5. Upsampling path
        for i, layer in enumerate(self.up_layers):
            # Skip connection
            skip_h = skip_connections[self.num_layers - 1 - i]
            h = torch.cat([h, skip_h], dim=1)
            h = layer(h)
        
        # 6. Final output
        out = self.final_proj(h)
        return out, h


class Denoising_Unet(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 activation, # Name of activation function
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 norm, # Name of norm function
                 skip_connections=True,
                 use_fast_version=False,  # NEW: Option to use fast version
                 ):
        super(Denoising_Unet, self).__init__()
        
        # Use fast version for inference speedup
        if use_fast_version:
            self.fast_net = FastDenoising_Unet(
                in_dim=in_dim, 
                num_hidden=num_hidden, 
                out_dim=out_dim,
                num_layers=min(num_layers, 2),  # Limit to 2 layers max
                activation=activation,
                feat_drop=feat_drop,
                norm=norm
            )
            self.use_fast = True
            return
        
        self.use_fast = False
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.activation = create_activation(activation)
        
        # --- Layer Definitions ---
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        # 1. Initial Projection Layer
        # This layer projects the input node features + time embedding into the hidden dimension
        # The time embedding will be added before this projection.
        self.initial_proj = nn.Linear(in_dim + num_hidden, num_hidden)

        # 2. Downsampling Path
        for _ in range(num_layers):
            self.down_layers.append(
                GATv2Conv(num_hidden, num_hidden // nhead, nhead,
                          feat_drop=feat_drop, attn_drop=attn_drop,
                          negative_slope=negative_slope,
                          activation=self.activation, # Activation can be part of the layer
                          allow_zero_in_degree=True, bias=True)
            )

        # 3. Bottleneck Layer
        self.middle_layer = GATv2Conv(num_hidden, num_hidden // nhead, nhead,
                                      feat_drop=feat_drop, attn_drop=attn_drop,
                                      negative_slope=negative_slope,
                                      activation=self.activation,
                                      allow_zero_in_degree=True, bias=True)

        # 4. Upsampling Path
        for _ in range(num_layers):
            # Input dimension is doubled because of the skip connection from the down path
            up_in_dim = num_hidden * 2 if skip_connections else num_hidden
            self.up_layers.append(
                GATv2Conv(up_in_dim, num_hidden // nhead, nhead,
                          feat_drop=feat_drop, attn_drop=attn_drop,
                          negative_slope=negative_slope,
                          activation=self.activation,
                          allow_zero_in_degree=True, bias=True)
            )

        # 5. Final Output Projection Layer
        # Projects the final hidden features back to the desired output dimension
        self.final_proj = nn.Linear(num_hidden, out_dim)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATv2Conv):
                # DGL's default initialization is usually sufficient (Xavier)
                pass

    def forward(self, g, x_t, time_embed):
        # Use fast version if enabled
        if self.use_fast:
            return self.fast_net(g, x_t, time_embed)
        
        # g: DGL graph
        # x_t: Noisy node features (N, in_dim)
        # time_embed: Time embeddings for each node (N, num_hidden)
        
        # 1. Initial Projection
        # Concatenate noisy features and time embedding
        h = torch.cat([x_t, time_embed], dim=1)
        h = self.initial_proj(h)
        h = self.activation(h)

        down_hidden_states = []

        # 2. Down Path
        for layer in self.down_layers:
            # Store the hidden state for skip connections
            down_hidden_states.append(h)
            h = layer(g, h).flatten(1)

        # 3. Bottleneck
        h = self.middle_layer(g, h).flatten(1)

        # 4. Up Path
        # Iterate in reverse to match the down path layers
        for i, layer in enumerate(self.up_layers):
            if self.skip_connections:
                # Concatenate with the corresponding hidden state from the down path
                skip_h = down_hidden_states[self.num_layers - 1 - i]
                h = torch.cat([h, skip_h], dim=1)
            
            h = layer(g, h).flatten(1)
        
        # 5. Final Output
        out = self.final_proj(h)

        # For compatibility with original DDM code, the second return value can be `h`
        # The original code expected concatenated hidden states, but that's not standard
        # and not used. Returning the final hidden state `h` is safer.
        return out, h


class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        # Ensure residual connection only happens if dimensions match
        res = x
        out = self.fnc(x, *args, **kwargs)
        if res.shape == out.shape:
            return out + res
        else:
            # This might happen if fnc changes dimensions unexpectedly.
            # In this specific U-Net, GAT output dim should match input dim.
            print(f"Warning: Residual connection shape mismatch: {res.shape} vs {out.shape}. Skipping residual.")
            return out


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'prelu'):
        super(MlpBlock, self).__init__()
        norm_layer = create_norm(norm)
        act_layer = create_activation(activation)

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # Using Sequential directly for the main path
        self.main_path = nn.Sequential(
            norm_layer(hidden_dim) if norm_layer else nn.Identity(),
            act_layer,
            nn.Linear(hidden_dim, hidden_dim),
            # Maybe add dropout here if needed: nn.Dropout(p=...)
            norm_layer(hidden_dim) if norm_layer else nn.Identity(),
            act_layer,
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_mlp = Residual(self.main_path) # Wrap main path in residual
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = act_layer # Use the same activation after output projection

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x) # Apply residual block
        x = self.out_proj(x)
        x = self.act(x)
        return x