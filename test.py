#
# newMain.py (Unsupervised GCN Pre-training + DDM Feature Enhancement)
#
import logging
import argparse
import os
import random
import warnings
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import scipy.sparse as sp
from tqdm import tqdm

# --- Project-specific Imports from First Script ---
from data import list_graph_loader, Datasets

# --- Evaluation Metrics ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import GraphDataLoader as DGLDataLoader
from sklearn.model_selection import train_test_split
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set
from dgl.nn.pytorch.conv import GraphConv
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", message="It is not recommended to directly access the internal storage format")

# =========================================================================
# STAGE 2: DDPM-BASED FEATURE ENHANCER (Adapted from your second script)
# =========================================================================
def data_split(list_adj, list_x, list_label, test_size=0.2, random_state=None):
    """
    Splits the dataset into training and testing sets using sklearn.
    This version is flexible and allows for reproducibility.
    """
    # Create indices from 0 to N-1
    indices = list(range(len(list_adj)))

    # Use stratify to ensure the same label distribution in train and test sets
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=list_label
    )

    # Use the generated indices to split your data lists
    list_adj_train = [list_adj[i] for i in train_indices]
    list_adj_test = [list_adj[i] for i in test_indices]

    list_x_train = [list_x[i] for i in train_indices]
    list_x_test = [list_x[i] for i in test_indices]

    list_label_train = [list_label[i] for i in train_indices]
    list_label_test = [list_label[i] for i in test_indices]

    return list_adj_train, list_adj_test, list_x_train, list_x_test, list_label_train, list_label_test

class EnhancedTimeEmbedding(nn.Module):
    """Enhanced time embedding with sinusoidal positional encoding + MLP projection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Simple embedding layer as a fallback
        self.embed = nn.Embedding(1000, dim)  # Direct embedding lookup for up to 1000 timesteps
        
        # Sinusoidal position embedding + learnable projection as main pathway
        self.time_proj = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Final MLP layer
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, t):
        # Handle both index tensor and already embedded tensor
        if t.dim() == 1:  # If t is just indices
            # First convert to float and normalize to [0,1]
            t_float = t.float() / 1000.0
            # Apply sinusoidal embedding
            t_emb = self.time_proj(t_float)
        else:  # If t is already embedded
            t_emb = t
        
        # Apply final MLP
        return self.time_mlp(t_emb)
        
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Ensure time has the right shape for broadcasting
        if time.dim() == 1:
            time = time.unsqueeze(1)  # [B] -> [B, 1]
            
        # Handle different input shapes
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Zero-pad if dim is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings

class ResidualGATBlock(nn.Module):
    """Residual GAT block with improved skip connections."""
    def __init__(self, in_dim, hidden_dim, nhead, feat_drop, attn_drop, negative_slope, norm, time_dim=None):
        super().__init__()
        self.use_time = time_dim is not None
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # Create normalization layers with explicit dimensions
        self.norm1 = nn.LayerNorm(in_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        
        # Time conditioning
        if self.use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, hidden_dim),
                nn.GELU()
            )
        
        # Main GAT layer
        self.gat = dgl.nn.GATv2Conv(
            in_dim, hidden_dim // nhead, nhead,
            feat_drop, attn_drop, negative_slope,
            False, activation=F.gelu, allow_zero_in_degree=True
        )
        
        # Output projection if dimensions don't match
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        
        # FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(feat_drop),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, g, x, time_emb=None):
        # Verify input dimensions match expected dimensions
        if x.shape[1] != self.in_dim:
            print(f"Warning: Input dimension mismatch in ResidualGATBlock. Expected {self.in_dim}, got {x.shape[1]}")
            # Adapt input dimensions using projection
            x = nn.Linear(x.shape[1], self.in_dim, device=x.device)(x)
            
        # First GAT block with residual
        h = self.norm1(x)
        h = self.gat(g, h).flatten(1)
        
        # Add time embedding if provided - expand to match node features
        if self.use_time and time_emb is not None:
            # Process the time embedding
            time_condition = self.time_mlp(time_emb)
            
            # Get batch structure to expand time embeddings
            batch_num_nodes = g.batch_num_nodes()
            
            # Expand time embeddings to match node features
            expanded_time_cond = []
            for i, num_nodes in enumerate(batch_num_nodes):
                # Repeat this graph's time embedding for all its nodes
                expanded_time_cond.append(time_condition[i].unsqueeze(0).expand(num_nodes, -1))
            
            # Stack all the expanded embeddings
            expanded_time_cond = torch.cat(expanded_time_cond, dim=0)
            
            # Now add the expanded time condition to node features
            h = h + expanded_time_cond
        
        # Apply residual connection
        h = h + self.proj(x)
        
        # FFN block with residual
        h2 = self.norm2(h)
        h2 = self.ffn(h2)
        return h + h2

class Denoising_Unet(nn.Module):
    """ Enhanced GATv2-based U-Net for the DDM with skip connections and better time conditioning. """
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, nhead, activation, feat_drop, attn_drop, negative_slope, norm):
        super(Denoising_Unet, self).__init__()
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.norm = norm
        
        # Better time embedding
        self.time_embedding = EnhancedTimeEmbedding(num_hidden)
        
        # Initial projection of input features - use dynamic sizing
        self.input_proj = None  # Will create dynamically in forward pass to ensure correct dimensions
        
        # Simplify architecture to use fixed dimensions throughout
        hidden_dim = num_hidden  # Use a consistent hidden dimension
        
        # Down blocks (encoder) - all using the same dimension
        self.down_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.down_blocks.append(ResidualGATBlock(
                hidden_dim, hidden_dim, nhead, feat_drop, attn_drop, negative_slope, norm, hidden_dim
            ))
        
        # Middle block
        self.middle_block = ResidualGATBlock(
            hidden_dim, hidden_dim, nhead, feat_drop, attn_drop, negative_slope, norm, hidden_dim
        )
        
        # Up blocks (decoder)
        self.up_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.up_blocks.append(ResidualGATBlock(
                hidden_dim, hidden_dim, nhead, feat_drop, attn_drop, negative_slope, norm, hidden_dim
            ))
            
        # Final output block
        self.final_res_block = ResidualGATBlock(
            hidden_dim, hidden_dim, nhead, feat_drop, attn_drop, negative_slope, norm, hidden_dim
        )
        
        self.output = nn.Sequential(
            nn.LayerNorm(num_hidden) if norm else nn.Identity(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, out_dim)
        )

    def forward(self, g, x_t, time_indices):
        # Print input shapes for debugging
        # print(f"Denoising_Unet input shapes: x_t={x_t.shape}, time_indices={time_indices.shape}")
        
        # Process time embedding - expect time_indices to be per-graph indices
        # Shape of time_indices should be [batch_size]
        if time_indices.dim() > 1:
            # If time_indices is already expanded, flatten it to batch dimension
            time_indices = time_indices.view(-1)[0:g.batch_size]
            
        t_emb = self.time_embedding(time_indices)
        # print(f"Time embedding shape: {t_emb.shape}")
        
        # Expand time embedding to match node features
        batch_num_nodes = g.batch_num_nodes()
        # print(f"Batch num nodes: {batch_num_nodes}, total: {sum(batch_num_nodes)}")
        expanded_t_emb = []
        
        # For each graph in the batch, repeat its time embedding for all nodes
        for i, num_nodes in enumerate(batch_num_nodes):
            if i < t_emb.shape[0]:  # Ensure we don't go out of bounds
                # Repeat this graph's time embedding for all its nodes
                expanded_t_emb.append(t_emb[i].unsqueeze(0).expand(num_nodes, -1))
            else:
                # Fallback if dimensions mismatch
                expanded_t_emb.append(torch.zeros(num_nodes, t_emb.shape[-1], device=t_emb.device))
        
        # Stack all the expanded embeddings
        expanded_t_emb = torch.cat(expanded_t_emb, dim=0)
        
        # Ensure dimensions match before concatenation
        if expanded_t_emb.shape[0] != x_t.shape[0]:
            # Log the dimension mismatch for debugging
            print(f"Warning: Time embedding dimension mismatch - x_t: {x_t.shape}, time_emb: {expanded_t_emb.shape}")
            # Adjust dimensions to match
            if expanded_t_emb.shape[0] > x_t.shape[0]:
                expanded_t_emb = expanded_t_emb[:x_t.shape[0]]
            else:
                # Pad with zeros to match
                padding = torch.zeros(x_t.shape[0] - expanded_t_emb.shape[0], 
                                     expanded_t_emb.shape[1], 
                                     device=expanded_t_emb.device)
                expanded_t_emb = torch.cat([expanded_t_emb, padding], dim=0)
        
        # Concatenate time embedding with node features
        h = torch.cat([x_t, expanded_t_emb], dim=1)
        
        # Create input projection dynamically based on actual input size
        input_dim = h.shape[1]
        target_dim = self.num_hidden
        # print(f"Creating input projection: {input_dim} -> {target_dim}")
        
        # Dynamic projection
        input_proj = nn.Sequential(
            nn.Linear(input_dim, target_dim, device=h.device),
            nn.GELU()
        )
        
        # Apply projection
        h = input_proj(h)
        
        # Down path with residuals - store intermediate features for skip connections
        skip_connections = [h]
        for down_block in self.down_blocks:
            h = down_block(g, h, t_emb)
            skip_connections.append(h)
        
        # Middle block processing
        h = self.middle_block(g, h, t_emb)
        
        # Up path - use residual connections without concatenation to avoid dimension issues
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections.pop()
            
            # Instead of concatenating, use additive skip connection
            # First project h to match skip dimensions if needed
            if h.shape[1] != skip.shape[1]:
                # Simple projection
                h = nn.Linear(h.shape[1], skip.shape[1], device=h.device)(h)
            
            # Add skip connection (residual)
            h = h + skip
            
            # Apply the block
            h = up_block(g, h, t_emb)
        
        # Final skip connection - simple addition instead of concatenation
        if h.shape[1] != skip_connections[0].shape[1]:
            h = nn.Linear(h.shape[1], skip_connections[0].shape[1], device=h.device)(h)
        
        h = h + skip_connections[0]
        
        # Final processing
        h = self.final_res_block(g, h, t_emb)
            
        # Final output
        denoised = self.output(h)
        return denoised, h

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        return torch.from_numpy(np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64))
    elif beta_schedule == "cosine":
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.from_numpy(np.clip(betas, 0.0001, 0.9999))
    else:
        raise NotImplementedError(beta_schedule)

def extract(v, t, x_shape):
    """Extracts coefficients at specified timesteps."""
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DDM(nn.Module):
    """ Enhanced Denoising Diffusion Model for graph feature learning with contrastive loss. """
    def __init__(self, in_dim, num_hidden, num_layers, nhead, activation, feat_drop, attn_drop, norm, **kwargs):
        super(DDM, self).__init__()
        self.T = kwargs.get('T', 1000)
        self.alpha_l = kwargs.get('alpha_l', 2.0)
        beta_schedule = kwargs.get('beta_schedule', 'cosine')
        beta_1 = kwargs.get('beta_1', 5e-5)  # Lower initial noise for better structure preservation
        beta_T = kwargs.get('beta_T', 0.02)  # End noise level
        
        # Get noise schedule
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, self.T)
        self.register_buffer('betas', beta)
        
        # Pre-compute diffusion values for q sampling
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        # Additional helper values for sampling and loss calculation
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1.0 / alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        
        # Main denoising U-Net
        self.net = Denoising_Unet(in_dim, num_hidden, in_dim, num_layers, nhead, activation, feat_drop, attn_drop, 0.2, norm)
        
        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden // 2)
        )

    def loss_fn(self, x, y):
        """Multi-component loss function with MSE, cosine similarity, and feature consistency."""
        
        # Cosine similarity loss for feature alignment
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        cos_loss = (1 - (x_norm * y_norm).sum(dim=-1)).pow_(self.alpha_l).mean()
        
        # Feature statistics alignment loss
        x_mean, x_std = x.mean(dim=0), x.std(dim=0) + 1e-6
        y_mean, y_std = y.mean(dim=0), y.std(dim=0) + 1e-6
        mean_loss = F.mse_loss(x_mean, y_mean)
        std_loss = F.mse_loss(x_std, y_std)
        
        return  cos_loss + 0.2 * (mean_loss + std_loss)

    def sample_q(self, t, x, batch_num_nodes=None):
        """Enhanced sampling for q(x_t | x_0) with structure-preserving noise."""
        # Compute mean and std per feature dimension
        miu, std = x.mean(dim=0), x.std(dim=0) + 1e-6
        
        # Generate and normalize noise
        noise = torch.randn_like(x, device=x.device)
        with torch.no_grad():
            # Normalize noise to match feature statistics
            noise = F.layer_norm(noise, (noise.shape[-1],))
            noise = noise * std + miu
            
            # Structure-preserving noise: preserve sign of original features
            noise = torch.sign(x) * torch.abs(noise)
            
            # Mix noise with original signal for early timesteps - simplified to avoid OOM
            # Calculate ratio as a scalar per timestep (batch)
            noise_ratio = torch.min(t.float() / (self.T * 0.1), torch.ones_like(t.float()))
            
            # Reshape to match batch structure (one scalar per graph)
            if batch_num_nodes is not None:
                # Create a list to hold per-node ratios
                batched_noise_ratio = []
                # For each graph in the batch
                for i, num_nodes in enumerate(batch_num_nodes):
                    # Repeat the ratio for all nodes in this graph
                    graph_ratio = noise_ratio[i].item()  # Get as scalar
                    batched_noise_ratio.extend([graph_ratio] * num_nodes)
                # Convert to tensor and add feature dimension
                noise_ratio = torch.tensor(batched_noise_ratio, device=x.device).unsqueeze(1)
            else:
                # Fallback for uniform batches
                batch_size = t.shape[0]
                num_nodes = x.shape[0]
                nodes_per_graph = num_nodes // batch_size
                # Create a tensor with repeated values
                noise_ratio = noise_ratio.repeat_interleave(nodes_per_graph).unsqueeze(1)
            
            # Memory-efficient mixing
            noise = noise * noise_ratio + x * (1 - noise_ratio)
    
        # Extract coefficients for each graph - shape will be [batch_size]
        sqrt_alphas_bar = torch.gather(self.sqrt_alphas_bar, index=t, dim=0).float()
        sqrt_one_minus_alphas_bar = torch.gather(self.sqrt_one_minus_alphas_bar, index=t, dim=0).float()
    
        # Repeat coefficients for each node based on batch structure
        if batch_num_nodes is not None:
            sqrt_alphas_bar = sqrt_alphas_bar.repeat_interleave(batch_num_nodes, dim=0)
            sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.repeat_interleave(batch_num_nodes, dim=0)
        else:
            # Fallback: assume uniform distribution
            batch_size = t.shape[0]
            num_nodes = x.shape[0]
            nodes_per_graph = num_nodes // batch_size
            sqrt_alphas_bar = sqrt_alphas_bar.repeat_interleave(nodes_per_graph, dim=0)
            sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.repeat_interleave(nodes_per_graph, dim=0)
    
        # Reshape for broadcasting - add dimension for features
        sqrt_alphas_bar = sqrt_alphas_bar.unsqueeze(-1)  # Shape: [num_nodes, 1]
        sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.unsqueeze(-1)  # Shape: [num_nodes, 1]
    
        # Apply the diffusion process
        x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * noise
        
        # Return noisy sample and original timestamps (not expanded per node)
        # The forward method will handle expanding these to match nodes
        return x_t, t, noise

    def forward(self, g, x):
        """Memory-efficient training forward pass with timestep sampling."""
        # Get batch structure information
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
        
        # print(f"DDM forward: batch_size={batch_size}, total nodes={x.shape[0]}")
    
        # Check if batch is too small - require at least 2 graphs for training
        if batch_size < 2:
            print(f"WARNING: Batch size {batch_size} is too small for training. Need at least 2 graphs.")
            return torch.tensor(0.0, requires_grad=True, device=x.device)
    
        # Use a simpler timestep sampling to reduce memory overhead
        # Uniform sampling across time steps with a bias toward later steps
        if torch.rand(1).item() > 0.7:
            # 30% of the time, use later timesteps (800-999)
            t = torch.randint(800, self.T, (batch_size,), device=x.device)
        else:
            # 70% of the time, use full range
            t = torch.randint(0, self.T, (batch_size,), device=x.device)
        
        # print(f"Sampled timesteps: {t}")
    
        try:
            # Generate noisy samples and denoising targets
            x_t, time_indices, _ = self.sample_q(t, x, batch_num_nodes)
        
            # Apply denoising network
            denoised_x, _ = self.net(g, x_t, time_indices)
            
            # Calculate loss
            loss = self.loss_fn(denoised_x, x)
            
            return loss
        except Exception as e:
            print(f"ERROR in DDM forward pass: {e}")
            # Return zero loss to avoid training crash, but with gradient
            return torch.tensor(0.0, requires_grad=True, device=x.device)

    @torch.no_grad()
    def embed(self, g, x, t_int):
        """Generate embeddings at a specific diffusion timestep."""
        # Get batch structure information
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
    
        # Create uniform timestamp tensor for evaluation
        t = torch.full((batch_size,), t_int, device=x.device, dtype=torch.long)
    
        # Generate noisy sample at specified timestep
        x_t, time_indices, _ = self.sample_q(t, x, batch_num_nodes)
    
        # Apply denoising network
        denoised_x, hidden_features = self.net(g, x_t, time_indices)
        
        # For feature extraction, we return both denoised features and intermediate representations
        return denoised_x




# --- Simple Supervised GCN model for Stage 1 ---
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim=1024, dropout=0.5, num_classes=1):  # Reduced hidden_dim, increased dropout
        super(SimpleGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        # First encoder layer: input to hidden
        self.encoder_layers.append(GraphConv(in_feats, hidden_dim, activation=F.relu))
        # Second encoder layer: hidden to output embedding
        self.encoder_layers.append(GraphConv(hidden_dim, out_feats, activation=F.relu))
        
        # Pooling functions for graph-level embeddings
        self.pool = AvgPooling()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, g, features):
        h = features
        
        # Apply GCN encoder layers
        for i, layer in enumerate(self.encoder_layers):
            h = self.dropout(h)  # Apply dropout to all layers
            h = layer(g, h)
        
        return h
        
    def get_graph_embedding(self, g, node_embeddings):
        # Pooling to get graph-level representation
        with g.local_scope():
            g.ndata['h'] = node_embeddings
            graph_embedding = self.pool(g, g.ndata['h'])
        
        return graph_embedding
    
    def forward(self, g, features):
        # Apply strong dropout to input
        features = self.dropout(features)
        
        # Encode
        node_embeddings = self.encode(g, features)
        
        # Get graph-level embedding
        graph_embedding = self.get_graph_embedding(g, node_embeddings)
        
        # Classification
        logits = self.classifier(graph_embedding)
        
        return node_embeddings, graph_embedding, logits


# --- Graph Transformer model for Stage 1 ---
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Define query, key, value projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.head_dim ** 0.5)
    
    def forward(self, g, h):
        """
        Args:
            g: DGL graph
            h: Node features (batch_size, num_nodes, in_dim)
        Returns:
            Updated node features (batch_size, num_nodes, out_dim)
        """
        batch_size = g.batch_size
        num_nodes = g.number_of_nodes() // batch_size
        
        # Reshape h from flattened graph to batched form
        h_reshaped = h.reshape(-1, self.head_dim)
        
        # Project inputs
        q = self.q_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: (batch_size, num_heads, num_nodes, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply attention mask based on graph structure
        src, dst = g.edges()
        edge_mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=h.device)
        
        # For each batch item, create adjacency mask
        for i in range(batch_size):
            # Get the nodes for this graph in the batch
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            
            # Get edges for this graph
            mask = (src >= start_idx) & (src < end_idx)
            graph_src = src[mask] - start_idx
            graph_dst = dst[mask] - start_idx
            
            # Set connections in the mask
            edge_mask[i, graph_src, graph_dst] = True
        
        # Add self-loops
        for i in range(batch_size):
            for j in range(num_nodes):
                edge_mask[i, j, j] = True
        
        # Apply the mask: large negative value for non-edges
        attn_mask = edge_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch_size, num_nodes, num_heads, head_dim)
        out = out.transpose(1, 2).contiguous()
        
        # Concatenate heads and project
        out = out.view(batch_size, num_nodes, self.out_dim)
        out = self.out_proj(out)
        
        # Flatten back to match DGL expected format
        return out.view(-1, self.out_dim)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, d_model, nhead, dropout)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.gelu
    
    def forward(self, g, x):
        # Self-attention block
        attn_output = self.self_attn(g, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim=1024, num_layers=3, nhead=8, 
                 dim_feedforward=2048, dropout=0.1, num_classes=1):
        super(GraphTransformer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Linear(in_feats, hidden_dim)
        
        # Position encoding
        self.position_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_feats)
        
        # Pooling functions for graph-level embeddings
        self.pool = AvgPooling()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_feats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_position(self, g):
        """Generate position encoding for each node based on graph structure."""
        # Get number of nodes per graph in the batch
        batch_num_nodes = g.batch_num_nodes()
        device = g.device
        
        position_embeddings = []
        
        for i, num_nodes in enumerate(batch_num_nodes):
            # Generate normalized position indices [0, 1]
            positions = torch.arange(num_nodes, device=device).float() / max(num_nodes - 1, 1)
            positions = positions.unsqueeze(1)  # Shape: [num_nodes, 1]
            
            # Encode positions
            pos_embed = self.position_encoder(positions)
            position_embeddings.append(pos_embed)
        
        # Concatenate for all nodes in batch
        return torch.cat(position_embeddings, dim=0)
    
    def encode(self, g, features):
        # Project features to hidden dimension
        h = self.embedding(features)
        
        # Add positional encoding
        pos_encoding = self.encode_position(g)
        h = h + pos_encoding
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(g, h)
        
        # Project to output dimension
        h = self.output_proj(h)
        
        return h
    
    def get_graph_embedding(self, g, node_embeddings):
        # Pooling to get graph-level representation
        with g.local_scope():
            g.ndata['h'] = node_embeddings
            graph_embedding = self.pool(g, g.ndata['h'])
        
        return graph_embedding
    
    def forward(self, g, features):
        # Apply dropout to input
        features = self.dropout(features)
        
        # Encode
        node_embeddings = self.encode(g, features)
        
        # Get graph-level embedding
        graph_embedding = self.get_graph_embedding(g, node_embeddings)
        
        # Classification
        logits = self.classifier(graph_embedding)
        
        return node_embeddings, graph_embedding, logits

# =========================================================================
# STAGE 2: EVALUATION AND HELPER FUNCTIONS
# =========================================================================

def collate_dgl(batch):
    graphs, labels = map(list, zip(*batch))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def evaluate_with_svm(model, data_loader, T_values, device):
    """Memory-efficient evaluation with simpler pooling and classifiers."""
    # Wrap everything in try/except to avoid crashes during evaluation
    try:
        model.eval()
        
        # Initialize basic pooling operations
        avg_pooler = AvgPooling()
        max_pooler = MaxPooling()
    
    # Get feature dimension from model - handle different model structures
        if hasattr(model.net, 'out_dim'):
            feature_dim = model.net.out_dim
        elif hasattr(model.net, 'num_hidden'):
            feature_dim = model.net.num_hidden
        else:
            # Default fallback
            feature_dim = 512
            
        # print(f"Using feature dimension: {feature_dim}")
        
        # Simpler attention pooling
        attn_gate_nn = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        ).to(device)
    except Exception as e:
        print(f"Error setting up attention pooling: {e}")
        # Simple fallback
        attn_gate_nn = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        ).to(device)
    attn_pooler = GlobalAttentionPooling(attn_gate_nn)
    
    # Set up Set2Set pooler with appropriate dimensions
    try:
        if hasattr(model.net, 'out_dim'):
            feature_dim = model.net.out_dim
        elif hasattr(model.net, 'num_hidden'):
            feature_dim = model.net.num_hidden
        else:
            feature_dim = 512   
        
        set2set_pooler = Set2Set(feature_dim, n_iters=3, n_layers=1).to(device)
    except Exception as e:
        print(f"Error setting up Set2Set pooler: {e}")
        # Create a dummy pooler that won't be used
        set2set_pooler = None
    
    # Storage for embeddings and labels
    all_labels, embeds_by_T = [], {t: [] for t in T_values}
    
    # Process each batch
    for batch_g, labels in tqdm(data_loader, desc="Extracting embeddings"):
        batch_g, labels = batch_g.to(device), labels.to(device)
        feat = batch_g.ndata['attr']
        all_labels.append(labels.cpu())
        
        # Process a subset of timesteps to save memory
        for t_val in T_values:
            # Clear cache between iterations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Get denoised node features at this timestep
            denoised_nodes = model.embed(batch_g, feat, t_val)
            
            # Apply pooling strategies
            avg_embed = avg_pooler(batch_g, denoised_nodes)
            max_embed = max_pooler(batch_g, denoised_nodes)
            attn_embed = attn_pooler(batch_g, denoised_nodes)
            
            # Use Set2Set pooler if available
            if set2set_pooler is not None:
                try:
                    set2set_embed = set2set_pooler(batch_g, denoised_nodes)
                    # Combine all embeddings
                    combined_embed = torch.cat([avg_embed, max_embed, attn_embed, set2set_embed], dim=-1)
                except Exception as e:
                    print(f"Set2Set pooling failed: {e}")
                    # Fall back to simpler pooling
                    combined_embed = torch.cat([avg_embed, max_embed, attn_embed], dim=-1)
            else:
                # Simpler pooling without Set2Set
                combined_embed = torch.cat([avg_embed, max_embed, attn_embed], dim=-1)
            
            # Store and immediately detach to free memory
            embeds_by_T[t_val].append(combined_embed.cpu().detach())
            
            # Explicitly delete tensors to free memory
            del denoised_nodes, avg_embed, max_embed, attn_embed, combined_embed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Process all labels and embeddings
    all_labels = torch.cat(all_labels, dim=0).numpy()
    final_embeds = {t: torch.cat(embeds, dim=0).detach().numpy() for t, embeds in embeds_by_T.items()}
    
    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Perform cross-validation evaluation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_list = []
    
    # For each fold in cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(final_embeds[T_values[0]], all_labels)):
        # print(f"Evaluating fold {fold_idx+1}/5...")
        test_scores = []
        
        # Use fewer timesteps for evaluation
        eval_subset = T_values[::2] if len(T_values) > 6 else T_values  # Take every other value
        
        # Process each timestep
        for t_val in eval_subset:
            x_train, x_test = final_embeds[t_val][train_idx], final_embeds[t_val][test_idx]
            y_train, y_test = all_labels[train_idx], all_labels[test_idx]
            
            # Simpler classifier set
            classifiers = [
                SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
                LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            ]
            
            # Train and get predictions
            t_scores = []
            for clf_idx, clf in enumerate(classifiers):
                try:
                    clf.fit(x_train, y_train)
                    if hasattr(clf, 'decision_function'):
                        scores = clf.decision_function(x_test)
                    else:
                        scores = clf.predict_proba(x_test)[:, 1]
                    t_scores.append(scores)
                except Exception as e:
                    print(f"Warning: Classifier {clf_idx} failed: {e}")
                    
            if t_scores:
                avg_score = np.mean(t_scores, axis=0)
                test_scores.append(avg_score)
        
        if test_scores:
            final_scores = np.mean(test_scores, axis=0)
            try:
                auc = roc_auc_score(y_test, final_scores)
                auc_list.append(auc)
                # print(f"Fold {fold_idx+1} AUC: {auc:.4f}")
            except Exception as e:
                print(f"Error computing AUC: {e}")
    
    if auc_list:
        return np.mean(auc_list), np.std(auc_list)
    else:
        print("WARNING: Evaluation failed completely!")
        return 0.0, 0.0


# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage Graph Classification: Unsupervised GCN Pre-training + DDM Enhancement')
    parser.add_argument('-e', dest="epoch_number", default=11, type=int)  # Reduced epochs
    parser.add_argument('-lr', dest="lr", default=1e-3, type=float)        # Higher learning rate
    parser.add_argument('-batchSize', dest="batchSize", default=32, type=int) # Larger batch for stability
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-graphEmDim', dest="graphEmDim", default=512, type=int) # Increased embedding dim
    parser.add_argument('-dataset', dest="dataset", default="Multi")
    parser.add_argument('-num_views', dest="num_views", default=2, type=int)
    args = parser.parse_args()

    # --- Setup ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    os.makedirs(f'ckpt/{args.dataset}', exist_ok=True)

    # --- Data Loading and Pre-processing ---
    print("--- Loading and Preparing Data ---")
    list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)
    
    list_adj_train, list_adj_test, list_x_train, list_x_test, list_label_train, list_label_test = data_split(list_adj, list_x, list_label, test_size=0.2, random_state=42)
    
    train_data = Datasets(list_adj_train, True, list_x_train, list_label_train)
    test_data = Datasets(list_adj_test, True, list_x_test, list_label_test, Max_num=train_data.max_num_nodes)
    train_data.processALL(self_for_none=True)
    test_data.processALL(self_for_none=True)
    in_feature_dim = train_data.feature_size

    def create_dgl_graphs(dataset_obj, num_views):
        dgl_graphs_list = []
        for i in range(len(dataset_obj.adj_s)):
            views = [dgl.from_scipy(sp.csr_matrix(dataset_obj.adj_s[i][v].cpu().numpy())) for v in range(num_views)]
            dgl_graphs_list.append(views)
        return dgl_graphs_list

    train_dgl_multiview = create_dgl_graphs(train_data, args.num_views)
    test_dgl_multiview = create_dgl_graphs(test_data, args.num_views)
    print(f"--- Data Loaded. Train: {len(train_data)}, Test: {len(test_data)} ---")

    # ========================================================================
    # STAGE 1: SUPERVISED GCN PRE-TRAINING
    # ========================================================================
    print("\n--- STAGE 1: Training Supervised Graph Transformer for Graph Embeddings ---")

    # Initialize the transformer model
    # Comment out the GCN model initialization
    """
    gcn_model = SimpleGCN(
        in_feats=in_feature_dim,
        out_feats=args.graphEmDim,  # Output embedding dimension
        dropout=0.3,
        num_classes=1  # Binary classification
    ).to(device)
    """

    # Initialize the transformer model instead
    transformer_model = GraphTransformer(
        in_feats=in_feature_dim,
        out_feats=args.graphEmDim,  # Output embedding dimension
        hidden_dim=1024,
        num_layers=3,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.2,
        num_classes=1  # Binary classification
    ).to(device)

    # # Supervised loss function
    # def supervised_loss_fn(logits, labels):
    #     # Binary cross entropy with logits
    #     return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())

    # # Optimizer with weight decay for regularization
    # optimizer_transformer = torch.optim.AdamW(
    #     transformer_model.parameters(), 
    #     lr=args.lr * 0.8,  # Slightly lower learning rate for transformer
    #     weight_decay=2e-4   # Higher regularization for transformer
    # )

    # # Learning rate scheduler with more aggressive early reduction
    # scheduler_transformer = ReduceLROnPlateau(
    #     optimizer_transformer, 
    #     mode='min', 
    #     factor=0.5, 
    #     patience=7,  # Reduced patience for faster LR adaptation
    #     verbose=True
    # )

    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 20  # Slightly reduced patience for early stopping
    checkpoint_dir = f"ckpt/{args.dataset}_stage1_transformer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_stage1_transformer_model.pt")

    # print("\n--- Starting Stage 1: Supervised Transformer Pre-training ---")
    # for epoch in range(args.epoch_number):
    #     transformer_model.train()
    #     train_data.shuffle()
        
    #     epoch_total_loss = 0
    #     all_train_preds = []
    #     all_train_labels = []
    #     num_batches = 0

    #     for i in range(0, len(train_data.list_adjs), args.batchSize):
    #         from_ = i
    #         to_ = i + args.batchSize
            
    #         # Get batch data with labels
    #         adj_batch, x_batch, _, _, _, labels_batch = train_data.get__(from_, to_, self_for_none=True, get_labels=True)
    #         target_labels = torch.tensor(labels_batch, dtype=torch.float32).to(device)

    #         # Prepare inputs for transformer model
    #         x_s_tensor = torch.stack(x_batch).to(device)
    #         features_for_dgl = x_s_tensor.view(-1, in_feature_dim)
            
    #         # Create a batch of graphs (using first view only)
    #         dgl_graphs = [dgl.from_scipy(sp.csr_matrix(g[0].cpu().numpy())).to(device) for g in adj_batch]
    #         batched_graph = dgl.batch(dgl_graphs)

    #         # Forward pass
    #         optimizer_transformer.zero_grad()
    #         node_embeddings, graph_embedding, logits = transformer_model(batched_graph, features_for_dgl)
            
    #         # Compute supervised loss
    #         loss = supervised_loss_fn(logits, target_labels)
            
    #         # Backward pass and optimization
    #         loss.backward()
    #         # Clip gradients to prevent explosion (important for transformers)
    #         torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
    #         optimizer_transformer.step()

    #         epoch_total_loss += loss.item()
    #         num_batches += 1
            
    #         # Store predictions for AUC calculation
    #         all_train_preds.append(torch.sigmoid(logits).detach().cpu())
    #         all_train_labels.append(target_labels.cpu())

    #     # End of epoch evaluation
    #     avg_loss = epoch_total_loss / num_batches
        
    #     # Calculate training AUC
    #     all_train_preds = torch.cat(all_train_preds).numpy().ravel()
    #     all_train_labels = torch.cat(all_train_labels).numpy().ravel()
    #     train_auc = roc_auc_score(all_train_labels, all_train_preds)
        
    #     # Evaluate on test set
    #     transformer_model.eval()
    #     test_total_loss = 0
    #     test_batches = 0
    #     all_test_preds = []
    #     all_test_labels = []
        
    #     with torch.no_grad():
    #         for i_test in range(0, len(test_data.list_adjs), args.batchSize):
    #             from_test = i_test
    #             to_test = i_test + args.batchSize
                
    #             adj_test, x_test, _, _, _, labels_test = test_data.get__(from_test, to_test, self_for_none=True, get_labels=True)
    #             target_labels_test = torch.tensor(labels_test, dtype=torch.float32).to(device)
                
    #             x_s_tensor_test = torch.stack(x_test).to(device)
    #             features_dgl_test = x_s_tensor_test.view(-1, in_feature_dim)
                
    #             # Create batch of test graphs (first view only)
    #             dgl_graphs_test = [dgl.from_scipy(sp.csr_matrix(g[0].cpu().numpy())).to(device) for g in adj_test]
    #             batched_graph_test = dgl.batch(dgl_graphs_test)
                
    #             # Forward pass with transformer
    #             node_embeddings, graph_embedding, logits = transformer_model(batched_graph_test, features_dgl_test)
                
    #             # Compute loss
    #             loss = supervised_loss_fn(logits, target_labels_test)
                
    #             test_total_loss += loss.item()
    #             test_batches += 1
                
    #             # Store predictions for AUC calculation
    #             all_test_preds.append(torch.sigmoid(logits).cpu())
    #             all_test_labels.append(target_labels_test.cpu())
        
    #     avg_test_loss = test_total_loss / test_batches
        
    #     # Calculate test AUC
    #     all_test_preds = torch.cat(all_test_preds).numpy().ravel()
    #     all_test_labels = torch.cat(all_test_labels).numpy().ravel()
    #     test_auc = roc_auc_score(all_test_labels, all_test_preds)
        
    #     # Update learning rate based on test loss
    #     scheduler_transformer.step(avg_test_loss)

    #     # Save model if test AUC improves
    #     if test_auc > best_val_auc:
    #         best_val_auc = test_auc
    #         best_val_loss = avg_test_loss
    #         best_epoch = epoch + 1
    #         patience_counter = 0
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': transformer_model.state_dict(),
    #             'optimizer_state_dict': optimizer_transformer.state_dict(),
    #             'test_loss': avg_test_loss,
    #             'test_auc': test_auc,
    #             'args': args
    #         }, best_model_path)
    #         print(f"  *** New best model saved! Best Test AUC: {best_val_auc:.4f} at epoch {best_epoch} ***")
    #     else:
    #         patience_counter += 1
            
    #     # Early stopping
    #     if patience_counter >= patience:
    #         print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
    #         break

    #     print(f"Epoch: {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Test Loss: {avg_test_loss:.4f} | Test AUC: {test_auc:.4f} | LR: {optimizer_transformer.param_groups[0]['lr']:.2e}")

    # print(f"--- STAGE 1 Finished. Supervised Transformer model is ready. Best Test AUC: {best_val_auc:.4f} ---")
    # # Clean up optimizer memory
    # del optimizer_transformer, scheduler_transformer
    # torch.cuda.empty_cache()

    # Load the best model
    if os.path.exists(best_model_path):
        print(f"Loading best Stage 1 model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']+1} with test AUC: {checkpoint['test_auc']:.4f}")
    else:
        print("WARNING: No best model checkpoint found. Using final model state.")


# ========================================================================
# BRIDGE: GENERATE DDM-READY DATASET FROM GCN EMBEDDINGS
# ========================================================================
print("\n--- BRIDGE: Generating DDM-ready dataset from Graph Transformer embeddings ---")
@torch.no_grad()
def generate_ddm_input_dataset(transformer_model, dataset_obj, dgl_multiview_list):
    transformer_model.eval()
    ddm_graphs = []

    # We process graph by graph to easily match embeddings with structures
    for i in tqdm(range(len(dataset_obj)), desc="Generating DDM Inputs"):
        # Prepare single-graph batch for transformer
        x = [dataset_obj.x_s[i].to_dense().clone().detach()]  # Fix tensor warning
        x_tensor = torch.stack(x).to(device)
        features_dgl = x_tensor.view(-1, in_feature_dim)
        
        # Merge multiple views into a single combined graph
        if len(dgl_multiview_list[i]) > 1:
            # Get all views
            views = [v.clone() for v in dgl_multiview_list[i]]
            
            # Create a combined graph with edges from all views
            # Start with the first view's graph
            combined_graph = views[0].clone()
            num_nodes = combined_graph.num_nodes()
            
            # Add edges from other views (union of all edges)
            for v_idx in range(1, min(args.num_views, len(views))):
                src, dst = views[v_idx].edges()
                combined_graph.add_edges(src, dst)
            
            # Add bidirectional edges to ensure information flow
            combined_graph = dgl.to_bidirected(combined_graph)
            combined_graph = combined_graph.to(device)
        else:
            # If only one view is available, use it directly
            combined_graph = dgl_multiview_list[i][0].to(device)
        
        # Process the combined graph through transformer
        node_embeddings, graph_embedding, _ = transformer_model(combined_graph, features_dgl)
        
        # Use the structure from the first view for the output graph
        dgl_graph_for_ddm = dgl_multiview_list[i][0]
        num_nodes = dgl_graph_for_ddm.num_nodes()

        # Calculate richer structural node properties
        # 1. Degrees
        in_degrees = dgl_graph_for_ddm.in_degrees().float().unsqueeze(1)
        out_degrees = dgl_graph_for_ddm.out_degrees().float().unsqueeze(1)
        total_degrees = (in_degrees + out_degrees)
        normalized_degrees = total_degrees / (total_degrees.max() + 1e-8)
        
        # 2. Approximation of node importance
        # Create undirected version for structural analysis
        g_undirected = dgl.to_bidirected(dgl_graph_for_ddm)
        
        # Calculate k-core approximation (using degree as proxy)
        kcore_proxy = normalized_degrees.clone()
        
        # 3. Node position as fraction of graph size
        position_encoding = torch.arange(0, num_nodes, device=device).float().unsqueeze(1) / max(1, num_nodes)
        
        # Use node embeddings from transformer model + graph-level context
        node_features = node_embeddings.clone()  # Start with node embeddings from transformer
        
        # Add graph-level information as global context (as a residual)
        graph_context = graph_embedding.expand(num_nodes, -1)
        node_features = node_features + 0.2 * graph_context  # Add graph context as residual
        
        # Create structural feature vector
        structural_features = torch.cat([
            normalized_degrees.to(device),
            kcore_proxy.to(device),
            position_encoding,
            # Add small noise component for diffusion model
            torch.randn(num_nodes, 1, device=device) * 0.01
        ], dim=1)
        
        # Project structural features to partial embedding dimension
        feature_dim = node_features.shape[-1]
        proj_dim = min(feature_dim // 4, 32)  # Use at most 1/4 of dimensions
        
        # Simple projection of structural features
        if structural_features.shape[1] < proj_dim:
            # Repeat to reach desired dimension
            structural_projected = structural_features.repeat(1, proj_dim // structural_features.shape[1] + 1)
            structural_projected = structural_projected[:, :proj_dim]  # Trim excess
        else:
            # Subsample
            structural_projected = structural_features[:, :proj_dim]
        
        # Scale structural features
        structural_weight = 0.2  # Balance between structure and semantic
        
        # Apply structural modifications to a portion of features
        if feature_dim >= proj_dim:
            node_features[:, :proj_dim] = node_features[:, :proj_dim] * (1 - structural_weight) + structural_projected * structural_weight
            
        # Add controlled noise
        uncertainty_scale = 0.05  # Fixed small uncertainty
        node_variation = torch.randn(num_nodes, feature_dim, device=device) * uncertainty_scale
        node_features = node_features + node_variation
        
        # Normalize features 
        node_features = F.layer_norm(node_features, (node_features.shape[-1],))
        
        # Store processed features
        dgl_graph_for_ddm.ndata['attr'] = node_features.cpu()
        ddm_graphs.append((dgl_graph_for_ddm, dataset_obj.labels[i]))
        
    return ddm_graphs
        
train_ddm_graphs = generate_ddm_input_dataset(transformer_model, train_data, train_dgl_multiview)
test_ddm_graphs = generate_ddm_input_dataset(transformer_model, test_data, test_dgl_multiview)

# ========================================================================
# STAGE 2: TRAIN DDM FEATURE ENHANCER
# ========================================================================
print(f"\n--- STAGE 2: Training DDM Feature Enhancer ---")
# Use a minimal batch size to avoid memory issues
memory_safe_batch_size = 8  # Force batch size to 2 regardless of user setting
print(f"Using minimal batch size {memory_safe_batch_size} for DDM training")
    
# Data loaders with minimal overhead
ddm_train_loader = DGLDataLoader(
    train_ddm_graphs, 
    batch_size=memory_safe_batch_size, 
    shuffle=True, 
    collate_fn=collate_dgl, 
    drop_last=True,
    num_workers=0  # No workers to avoid memory duplication
)
ddm_test_loader = DGLDataLoader(
    test_ddm_graphs, 
    batch_size=memory_safe_batch_size, 
    shuffle=False, 
    collate_fn=collate_dgl,
    num_workers=0  # No workers to avoid memory duplication
)
full_train_loader = DGLDataLoader(
    train_ddm_graphs,
    batch_size=len(train_ddm_graphs),
    shuffle=False,
    collate_fn=collate_dgl,
    num_workers=0
)

# Replace full_train_loader to use test_ddm_graphs
full_test_loader = DGLDataLoader(
    test_ddm_graphs,
    batch_size=len(test_ddm_graphs),
    shuffle=False,
    collate_fn=collate_dgl,
    num_workers=0
)

# Enhanced model configuration with increased capacity
ddm_main_args = {
    'in_dim': args.graphEmDim,
    'num_hidden': 512,    # Larger hidden dimension
    'num_layers': 2,       # Deeper network for more capacity
    'nhead': 8,            # Keep 8 attention heads for expressiveness
    'activation': 'gelu',  # GELU for better gradients
    'feat_drop': 0.1,      # Modest dropout for regularization
    'attn_drop': 0.1,      # Match with feat_drop
    'norm': 'layernorm'    # Layer normalization for stability
}
    
# Better diffusion parameters
ddm_kwargs = {
    'T': 1000,              # Keep 1000 diffusion steps
    'beta_schedule': 'cosine',  # Cosine schedule is smoother
    'alpha_l': 2.0,        # Stronger consistency loss
    'beta_1': 5e-5,        # Smaller starting noise 
    'beta_T': 0.02         # Keep end noise the same
}
    
print("Initializing DDM with enhanced architecture...")
ddm_model = DDM(**ddm_main_args, **ddm_kwargs).to(device)
    
# Add math import needed for enhanced time embeddings
import math
    
# Optimizer with larger learning rate and weight decay
optimizer_ddm = torch.optim.AdamW(
    ddm_model.parameters(),
    lr=5e-5,             # Slightly higher learning rate
    weight_decay=2e-4,   # More regularization
    betas=(0.9, 0.999)   # Standard betas
)
    
# Learning rate warmup + cosine decay for better convergence
total_steps = 1000  # Maximum epochs
warmup_steps = 50   # Warmup over first 50 epochs
    
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))  # Cosine decay
    
scheduler_ddm = LambdaLR(optimizer_ddm, lr_lambda)

# Evaluation setup with more sophisticated timestep selection
best_val_auc = 0.0
    
# Use fewer evaluation timesteps to reduce memory usage
eval_T_values = [100, 500, 900]
    
ddm_patience = 0
ddm_patience_limit = 200  # Reduced patience since we do more thorough evaluations
    
# EMA model for more stable results (exponential moving average)
ema_model = None
ema_decay = 0.995  # High decay for stability
    
# Function to update EMA model
def update_ema_model(current_model, ema_model, decay):
    if ema_model is None:
        ema_model = copy.deepcopy(current_model)
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model
            
    with torch.no_grad():
        for ema_param, current_param in zip(ema_model.parameters(), current_model.parameters()):
            ema_param.data.mul_(decay).add_(current_param.data, alpha=1 - decay)
    return ema_model
    
# Early stopping setup with both loss and AUC monitoring
best_train_loss = float('inf')
loss_plateau_count = 0
loss_plateau_threshold = 10  # Number of epochs with no improvement
loss_improvement_threshold = 1e-4  # Minimum improvement to reset counter
    
print("Starting DDM training with enhanced schedule...")
for epoch in range(1):  # More epochs for DDM
    ddm_model.train()
    total_loss = 0
        
    # Add progress bar for better monitoring
    progress_bar = tqdm(ddm_train_loader, desc=f"Epoch {epoch+1} Training")
        
    for batch_g, _ in progress_bar:
        batch_g = batch_g.to(device)
        features = batch_g.ndata['attr']
            
        # Zero gradients
        optimizer_ddm.zero_grad()
            
        # Forward pass with model
        loss = ddm_model(batch_g, features)
            
        # Backward pass and optimization
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(ddm_model.parameters(), max_norm=1.0)
        optimizer_ddm.step()
            
        # Track loss
        current_loss = loss.item()
        total_loss += current_loss
            
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
    # Update learning rate scheduler
    scheduler_ddm.step()
    avg_loss = total_loss / len(ddm_train_loader)
        
    # Update EMA model
    ema_model = update_ema_model(ddm_model, ema_model, ema_decay)
        
    # Monitor for loss plateaus
    if avg_loss < best_train_loss - loss_improvement_threshold:
        best_train_loss = avg_loss
        loss_plateau_count = 0
    else:
        loss_plateau_count += 1
        
    # Evaluate every 5 epochs to save computation
    if epoch % 5 == 0 or epoch >= 300 or epoch < 30:
        # Switch to EMA model for evaluation after it's initialized
        eval_model = ema_model if ema_model is not None and epoch > 50 else ddm_model
        eval_model.eval()
            
        # Enhanced evaluation with more focus on model quality
        auc_mean, auc_std = evaluate_with_svm(eval_model, ddm_test_loader, eval_T_values, device)
        print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | Test AUC (SVM): {auc_mean:.4f} ± {auc_std:.4f} | LR: {optimizer_ddm.param_groups[0]['lr']:.2e}")
            
        if auc_mean > best_val_auc:
            improvement = auc_mean - best_val_auc
            best_val_auc = auc_mean
            ddm_patience = 0
            print(f"*** New best test AUC: {best_val_auc:.4f} (↑{improvement:.4f}). Saving model. ***")
                
            # Save both regular and EMA model
            torch.save({
                'model_state_dict': ddm_model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state_dict': optimizer_ddm.state_dict(),
                'scheduler_state_dict': scheduler_ddm.state_dict(),
                'epoch': epoch,
                'best_auc': best_val_auc,
                'eval_T_values': eval_T_values
            }, f'ckpt/{args.dataset}/best_ddm_model.pth')
        else:
            ddm_patience += 1
                
        # Early stopping conditions - either patience reached or loss plateaued
        if ddm_patience >= ddm_patience_limit:
            print(f"Early stopping DDM training at epoch {epoch+1} due to no improvement in AUC")
            break
                
        if loss_plateau_count >= loss_plateau_threshold and epoch > 200:
            print(f"Early stopping DDM training at epoch {epoch+1} due to loss plateau")
            break



print(f"\n--- Pipeline Finished. Best Test AUC with SVM: {best_val_auc:.4f} ---")

# =========================================================================
# VISUALIZING EMBEDDINGS USING t-SNE
# =========================================================================
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
del optimizer_ddm, scheduler_ddm, ddm_train_loader, ddm_test_loader
import gc
gc.collect()
torch.cuda.empty_cache()

@torch.no_grad()
def extract_embeddings(model, data_loader, t_val, device):
    """Extract graph-level embeddings from the model."""
    model.eval()
    
    # Initialize basic pooling operations
    avg_pooler = AvgPooling()
    max_pooler = MaxPooling()
    
    # Storage for embeddings and labels
    all_embeds = []
    all_labels = []
    
    # Process each batch
    for batch_g, labels in tqdm(data_loader, desc="Extracting embeddings for t-SNE"):
        batch_g, labels = batch_g.to(device), labels.to(device)
        feat = batch_g.ndata['attr']
        all_labels.append(labels.cpu())
        
        # Get denoised node features at this timestep
        denoised_nodes = model.embed(batch_g, feat, t_val)
        
        # Apply pooling strategies
        avg_embed = avg_pooler(batch_g, denoised_nodes)
        max_embed = max_pooler(batch_g, denoised_nodes)
        
        # Combine embeddings
        combined_embed = torch.cat([avg_embed, max_embed], dim=-1)
        
        # Store embeddings
        all_embeds.append(combined_embed.cpu().detach())
    
    # Process all labels and embeddings
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_embeds = torch.cat(all_embeds, dim=0).detach().numpy()
    
    return all_embeds, all_labels

def visualize_tsne(embeddings, labels, perplexity=30, n_iter=1000, title="t-SNE Visualization of Graph Embeddings"):
    """Apply t-SNE and visualize embeddings."""
    print(f"Running t-SNE with perplexity={perplexity}, max_iter={n_iter}...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7
        )
    
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'ckpt/{args.dataset}/tsne_visualization.png', dpi=300)
    print(f"t-SNE visualization saved to ckpt/{args.dataset}/tsne_visualization.png")
    
    # Display if in interactive environment
    plt.show()

print("\n--- Generating t-SNE Visualization ---")

# Load the best DDM model for visualization
try:
    best_model_path = f'ckpt/{args.dataset}/best_ddm_model.pth'
    print(f"Loading best model from {best_model_path}")
    # Using weights_only=False for PyTorch 2.6 compatibility
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Try loading EMA model first, fallback to regular model
    if checkpoint['ema_model_state_dict'] is not None:
        print("Using EMA model for visualization")
        ddm_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("Using regular model for visualization")
        ddm_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Use middle timestep for visualization (not too noisy, not too clean)
    t_val = 500
    print(f"Extracting embeddings at timestep t={t_val}...")
    
    # Extract embeddings from test set
    embeddings, labels = extract_embeddings(ddm_model, full_test_loader, t_val, device)
    
    # Visualize using t-SNE
    print(f"Shape of embeddings: {embeddings.shape}, labels: {labels.shape}")
    visualize_tsne(embeddings, labels, perplexity=min(30, len(labels)-1), 
                  title=f"t-SNE Visualization of DDM Embeddings (t={t_val})")
    
    # Add multi-timestep visualization option
    print("\n--- Extracting Embeddings at Multiple Timesteps ---")
    # Sample a few timesteps to compare
    timesteps_to_visualize = [100, 500, 900]  # Early, middle, late
    multi_timestep_embeds = {}
    
    # Extract embeddings at different timesteps
    for ts in timesteps_to_visualize:
        print(f"Extracting embeddings at timestep {ts}...")
        ts_embeds, _ = extract_embeddings(ddm_model, full_test_loader, ts, device)
        multi_timestep_embeds[ts] = ts_embeds
    
    # Add PCA visualization option
    from sklearn.decomposition import PCA
    
    def visualize_pca(embeddings, labels, n_components=2, title="PCA Visualization of Graph Embeddings"):
        """Apply PCA and visualize embeddings."""
        print(f"Running PCA with n_components={n_components}...")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        embeddings_2d = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_.sum()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7
            )
        
        plt.title(f"{title}\nExplained Variance: {explained_variance:.2%}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        output_filename = f'ckpt/{args.dataset}/pca_visualization_{title.split()[1].lower()}.png'
        plt.savefig(output_filename, dpi=300)
        print(f"PCA visualization saved to {output_filename}")
        plt.show()
    
    def visualize_timestep_comparison(multi_ts_embeds, labels, method='tsne', perplexity=30, n_iter=1000):
        """Create a comparison plot of embeddings at different timesteps."""
        timesteps = list(multi_ts_embeds.keys())
        n_timesteps = len(timesteps)

        # Create a figure with n_timesteps plots in one row
        plt.figure(figsize=(n_timesteps * 6, 6))

        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, ts in enumerate(timesteps):
            plt.subplot(1, n_timesteps, i+1)

            if method.lower() == 'tsne':
                # Apply t-SNE
                tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
                reduced_embeds = tsne.fit_transform(multi_ts_embeds[ts])
                method_name = 't-SNE'
            else:
                # Apply PCA
                pca = PCA(n_components=2)
                reduced_embeds = pca.fit_transform(multi_ts_embeds[ts])
                explained_var = pca.explained_variance_ratio_.sum()
                method_name = f'PCA (Var: {explained_var:.2%})'
            
            # Plot each class
            for j, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(
                    reduced_embeds[mask, 0], 
                    reduced_embeds[mask, 1],
                    c=[colors[j]],
                    label=f'Class {label}',
                    alpha=0.7
                )
            
            plt.title(f"Timestep {ts} ({method_name})")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            
            # Only show legend on the first plot to save space
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        output_filename = f'ckpt/{args.dataset}/timestep_comparison_{method.lower()}.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Timestep comparison visualization saved to {output_filename}")
        plt.show()
    
    # Visualize embeddings at different timesteps
    print("\n--- Visualizing Embeddings at Different Timesteps ---")
    visualize_timestep_comparison(multi_timestep_embeds, labels, method='tsne')
    visualize_timestep_comparison(multi_timestep_embeds, labels, method='pca')
    
    # Also try PCA for the main embeddings
    print("\n--- Visualizing with PCA instead of t-SNE ---")
    visualize_pca(embeddings, labels, title=f"PCA of DDM Embeddings")
    
    # Now also visualize embeddings from the Graph Transformer
    print("\n--- Extracting Graph Transformer Embeddings ---")
    
    @torch.no_grad()
    def extract_transformer_embeddings(model, dataset_obj, dgl_multiview_list):
        """Extract graph-level embeddings from the Graph Transformer."""
        model.eval()
        all_embeds = []
        all_labels = []
        
        # We process graph by graph
        for i in tqdm(range(len(dataset_obj)), desc="Extracting Transformer Embeddings"):
            # Prepare single-graph batch for transformer
            x = [dataset_obj.x_s[i].to_dense().clone().detach()]
            x_tensor = torch.stack(x).to(device)
            features_dgl = x_tensor.view(-1, in_feature_dim)
            
            # Get the graph
            combined_graph = dgl_multiview_list[i][0].to(device)
            
            # Process through transformer
            _, graph_embedding, _ = transformer_model(combined_graph, features_dgl)
            
            # Store embeddings and labels
            all_embeds.append(graph_embedding.cpu())
            all_labels.append(dataset_obj.labels[i])
        
        # Convert to numpy arrays
        all_embeds = torch.cat(all_embeds, dim=0).numpy()
        all_labels = np.array(all_labels)
        
        return all_embeds, all_labels
    
    # Extract transformer embeddings from test set
    transformer_embeds, transformer_labels = extract_transformer_embeddings(
        transformer_model, test_data, test_dgl_multiview
    )
    
    # Visualize transformer embeddings
    print(f"Shape of transformer embeddings: {transformer_embeds.shape}")
    visualize_tsne(transformer_embeds, transformer_labels, 
                  perplexity=min(30, len(transformer_labels)-1),
                  title="t-SNE Visualization of Graph Transformer Embeddings")
    visualize_pca(transformer_embeds, transformer_labels, title="PCA of Transformer")
    
    # Compare both embeddings in a single plot
    print("\n--- Creating Combined Visualization ---")
    
    def evaluate_embedding_quality(embeddings, labels):
        """Evaluate embeddings using clustering metrics."""
        try:
            # Calculate silhouette score (higher is better)
            sil_score = silhouette_score(embeddings, labels)
            
            # Calculate Davies-Bouldin index (lower is better)
            db_score = davies_bouldin_score(embeddings, labels)
            
            print(f"Embedding Quality Metrics:")
            print(f"  Silhouette Score: {sil_score:.4f} (higher is better)")
            print(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")
            
            return sil_score, db_score
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
            return None, None
    
    # Evaluate embedding quality
    print("\n--- Evaluating Graph Transformer Embeddings ---")
    transformer_sil, transformer_db = evaluate_embedding_quality(transformer_embeds, transformer_labels)
    
    print("\n--- Evaluating DDM Enhanced Embeddings ---")
    ddm_sil, ddm_db = evaluate_embedding_quality(embeddings, labels)
    
    def visualize_comparison(ddm_embeds, transformer_embeds, labels, perplexity=30, n_iter=1000):
        """Create a comparison plot of both embedding types."""
        plt.figure(figsize=(18, 8))
    
    # First, visualize transformer embeddings
        plt.subplot(1, 2, 1)
        tsne_transformer = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
        transformer_2d = tsne_transformer.fit_transform(transformer_embeds)
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                transformer_2d[mask, 0], 
                transformer_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7
            )
        
        plt.title("Graph Transformer Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        
            # Then, visualize DDM embeddings
        plt.subplot(1, 2, 2)
        tsne_ddm = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
        ddm_2d = tsne_ddm.fit_transform(ddm_embeds)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                ddm_2d[mask, 0], 
                ddm_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7
            )
        
        plt.title(f"DDM Enhanced Embeddings (t={t_val})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'ckpt/{args.dataset}/tsne_comparison.png', dpi=300)
        print(f"Comparison visualization saved to ckpt/{args.dataset}/tsne_comparison.png")
        plt.show()
    
    # Create comparison visualization
    visualize_comparison(embeddings, transformer_embeds, labels, perplexity=min(30, len(labels)-1))
    
    # Add K-means clustering visualization
   

    def visualize_kmeans_clusters(embeddings, true_labels, n_clusters=None, title="K-means Clustering"):
        """Visualize K-means clustering results compared to true labels."""
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: True labels
        unique_labels = np.unique(true_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = true_labels == label
            ax1.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7
            )
        
        ax1.set_title("True Labels")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.legend()
        
        # Plot 2: K-means clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax2.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[cluster_colors[i]],
                label=f'Cluster {i}',
                alpha=0.7
            )
        
        # Skip drawing cluster centers since TSNE doesn't have a transform method
        # Instead, calculate approximate positions based on closest points
        centers_approx = []
        for i in range(n_clusters):
            # Find points in this cluster
            mask = cluster_labels == i
            if np.any(mask):
                # Use the mean position of the points in the cluster as approximation
                center_approx = np.mean(embeddings_2d[mask], axis=0)
                centers_approx.append(center_approx)
        
        # Draw approximate centers if we have any
        if centers_approx:
            centers_approx = np.array(centers_approx)
            ax2.scatter(
                centers_approx[:, 0], centers_approx[:, 1],
                marker='X', s=150, c='black', alpha=1.0,
                label='Approx. Centroids'
            )
        
        ax2.set_title(f"K-means Clusters (k={n_clusters})")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        output_filename = f'ckpt/{args.dataset}/kmeans_clustering_{title.split()[0].lower()}.png'
        plt.savefig(output_filename, dpi=300)
        print(f"K-means clustering visualization saved to {output_filename}")
        plt.show()
        
        return cluster_labels

    # Visualize clustering for both embedding types
    print("\n--- Visualizing K-means Clustering on Embeddings ---")
    ddm_clusters = visualize_kmeans_clusters(embeddings, labels, 
                                           title=f"DDM Embeddings (t={t_val})")
    transformer_clusters = visualize_kmeans_clusters(transformer_embeds, transformer_labels,
                                                    title="Transformer Embeddings")

    # Save all visualizations to a single PDF report
    def save_visualization_report():
        """Save all visualizations to a single PDF report."""
        output_filename = f'ckpt/{args.dataset}/embedding_visualization_report.pdf'
        
        with PdfPages(output_filename) as pdf:
            # Add a title page
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, f"Graph Embedding Visualization Report\nDataset: {args.dataset}",
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=20, transform=plt.gca().transAxes)
            pdf.savefig()
            plt.close()
            
            # Add clustering metrics table
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis('off')
            table_data = [
                ['Embedding Type', 'Silhouette Score ↑', 'Davies-Bouldin Index ↓'],
                ['Graph Transformer', f"{transformer_sil:.4f}", f"{transformer_db:.4f}"],
                ['DDM Enhanced', f"{ddm_sil:.4f}", f"{ddm_db:.4f}"]
            ]
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            for i in range(len(table_data)):
                for j in range(len(table_data[0])):
                    cell = table[i, j]
                    if i == 0:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor('#4472C4')
                    elif j == 0:
                        cell.set_text_props(weight='bold')
                        if i % 2 == 1:
                            cell.set_facecolor('#D9E1F2')
                        else:
                            cell.set_facecolor('#E9EDF4')
                    else:
                        if i % 2 == 1:
                            cell.set_facecolor('#D9E1F2')
                        else:
                            cell.set_facecolor('#E9EDF4')
                            
            plt.title('Embedding Quality Metrics Comparison', fontsize=16, pad=20)
            pdf.savefig()
            plt.close()
            
            # Add existing figures
            figures_path = f'ckpt/{args.dataset}/'
            for filename in ['tsne_comparison.png', 'timestep_comparison_tsne.png', 'timestep_comparison_pca.png']:
                try:
                    plt.figure(figsize=(10, 8))
                    img = plt.imread(figures_path + filename)
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                except Exception as e:
                    print(f"Couldn't add {filename} to report: {e}")
            
        print(f"Complete visualization report saved to {output_filename}")

    # Generate the report
    save_visualization_report()

except Exception as e:
    print(f"Error during t-SNE visualization: {e}")

def evaluate_clustering_with_ari_nmi(embeddings, true_labels, n_clusters=None, name="Embedding"):
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    print(f"{name} Clustering Quality:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f} (higher is better)")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f} (higher is better)")
    return ari, nmi

# === Run ARI/NMI evaluation instead of t-SNE ===
print("\n--- Evaluating DDM Embeddings with ARI/NMI ---")
try:
    best_model_path = f'ckpt/{args.dataset}/best_ddm_model.pth'
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if checkpoint['ema_model_state_dict'] is not None:
        print("Using EMA model for evaluation")
        ddm_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("Using regular model for evaluation")
        ddm_model.load_state_dict(checkpoint['model_state_dict'])
    t_val = 500
    print(f"Extracting embeddings at timestep t={t_val}...")
    embeddings, labels = extract_embeddings(ddm_model, full_test_loader, t_val, device)
    print(f"Shape of embeddings: {embeddings.shape}, labels: {labels.shape}")
    evaluate_clustering_with_ari_nmi(embeddings, labels, name=f"DDM Embeddings (t={t_val})")
    print("\n--- Evaluating Graph Transformer Embeddings with ARI/NMI ---")
    transformer_embeds, transformer_labels = extract_transformer_embeddings(
        transformer_model, test_data, test_dgl_multiview
    )
    evaluate_clustering_with_ari_nmi(transformer_embeds, transformer_labels, name="Graph Transformer Embeddings")
except Exception as e:
    print(f"Error during ARI/NMI evaluation: {e}")


def visualize_tsne(embeddings, labels, title="t-SNE Visualization", save_path=None):
    """Visualize high-dimensional embeddings in 2D using t-SNE."""
    tsne = TSNE(n_components=2, perplexity=min(30, len(labels)-1), max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7
        )
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE plot saved to {save_path}")
    plt.show()

# For DDM embeddings
visualize_tsne(embeddings, labels, title=f"t-SNE of DDM Embeddings (t={t_val})", save_path=f'ckpt/{args.dataset}/tsne_ddm_plotONly.png')

# For Transformer embeddings
visualize_tsne(transformer_embeds, transformer_labels, title="t-SNE of Graph Transformer Embeddings", save_path=f'ckpt/{args.dataset}/tsne_transformer_plotOnly.png')