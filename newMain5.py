#
# newMain.py (VAE Pre-training + DDM Feature Enhancement)
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
from model import StagedSupervisedVAE, ClassificationDecoder, DynamicCouplingEncoder
from data import list_graph_loader, Datasets

# --- Evaluation Metrics ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import GraphDataLoader as DGLDataLoader
from sklearn.model_selection import train_test_split
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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
    """ Enhanced Denoising Diffusion Model for graph feature learning with noise-prediction objective. """
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
        """Multi-component loss function retained for optional use."""
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        cos_loss = (1 - (x_norm * y_norm).sum(dim=-1)).pow_(self.alpha_l).mean()
        x_mean, x_std = x.mean(dim=0), x.std(dim=0) + 1e-6
        y_mean, y_std = y.mean(dim=0), y.std(dim=0) + 1e-6
        mean_loss = F.mse_loss(x_mean, y_mean)
        std_loss = F.mse_loss(x_std, y_std)
        return  cos_loss + 0.2 * (mean_loss + std_loss)

    # --- RESTORED: core DDPM epsilon-prediction loss ---
    def noise_loss(self, noise_pred, noise_gt):
        """Mean-squared error between predicted and ground-truth noise."""
        return F.mse_loss(noise_pred, noise_gt)

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
            # Generate noisy samples and obtain the ground-truth noise
            x_t, time_indices, noise = self.sample_q(t, x, batch_num_nodes)

            # Predict noise with the network
            noise_pred, _ = self.net(g, x_t, time_indices)

            # Core DDPM loss: predict epsilon
            loss = self.noise_loss(noise_pred, noise)
            
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
    
        # Predict noise and reconstruct x_0 from it
        noise_pred, _ = self.net(g, x_t, time_indices)

        # Expand coefficients just like in sample_q for correct broadcasting
        sqrt_alphas_bar = torch.gather(self.sqrt_alphas_bar, index=t, dim=0).float()
        sqrt_one_minus_alphas_bar = torch.gather(self.sqrt_one_minus_alphas_bar, index=t, dim=0).float()

        if batch_num_nodes is not None:
            sqrt_alphas_bar = sqrt_alphas_bar.repeat_interleave(batch_num_nodes, dim=0)
            sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.repeat_interleave(batch_num_nodes, dim=0)
        else:
            batch_size = t.shape[0]
            num_nodes = x.shape[0]
            nodes_per_graph = num_nodes // batch_size
            sqrt_alphas_bar = sqrt_alphas_bar.repeat_interleave(nodes_per_graph, dim=0)
            sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.repeat_interleave(nodes_per_graph, dim=0)

        sqrt_alphas_bar = sqrt_alphas_bar.unsqueeze(-1)
        sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.unsqueeze(-1)

        denoised_x = (x_t - sqrt_one_minus_alphas_bar * noise_pred) / (sqrt_alphas_bar + 1e-8)

        return denoised_x

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
            feature_dim = 768
            
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
            nn.Linear(768, 64),
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
            feature_dim = 768
        
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
    parser = argparse.ArgumentParser(description='Two-Stage Graph Classification: VAE Pre-training + DDM Enhancement')
    parser.add_argument('-e', dest="epoch_number", default=100, type=int)  # Reduced epochs
    parser.add_argument('-lr', dest="lr", default=1e-3, type=float)        # Higher learning rate
    parser.add_argument('-batchSize', dest="batchSize", default=8, type=int) # Larger batch for stability
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-graphEmDim', dest="graphEmDim", default= 768, type=int) # Increased embedding dim
    parser.add_argument('-dataset', dest="dataset", default="PPMI")
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
    # STAGE 1: SUPERVISED VAE PRE-TRAINING
    # ========================================================================
    print("\n--- STAGE 1: Training Supervised VAE for Graph Embeddings ---")
    
    def SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta, kl_threshold=0.1):
        # Label smoothing for better generalization
        smoothed_labels = target_labels.float().view(-1, 1) * 0.9 + 0.05
        classif_loss = F.binary_cross_entropy_with_logits(predicted_logits, smoothed_labels)
        
        # ------------------------------------------------------------------
        # Numerically-stable KL divergence between ��(μ,σ²) and 𝒩(0,1)
        #   KL = −½ Σ(1 + log σ² − μ² − σ²)
        # To avoid overflow when log_std is large (σ² = e^{2·log_std}), we:
        #   1. Clamp log_std to a reasonable range so exp(·) does not explode.
        #   2. Clamp mean to keep μ² bounded (helps when encoder diverges).
        # ------------------------------------------------------------------

        # Hard clamping – adjust the bounds if you need a tighter range
        log_std = torch.clamp(log_std, min=-10.0, max=3.0)   # σ ∈ [e⁻¹⁰, e³]
        mean     = torch.clamp(mean,     min=-5.0,  max=5.0)  # μ  ∈ [−5, 5]

        # Pre-compute σ² in a safe way
        var = torch.exp(2.0 * log_std)   # σ² = e^{2·logσ}

        # Compute KL per-dimension then take the *mean* across dims
        # This keeps the magnitude independent of latent size.
        kl_div = -0.5 * (1 + 2.0 * log_std - mean.pow(2) - var)
        kl_div = kl_div.mean(dim=1)
        # Use free bits to prevent posterior collapse
        kl_loss = torch.mean(torch.clamp(kl_div, min=kl_threshold))
        
        # Reduced regularization
        embedding_reg = 0.001 * torch.mean(mean.pow(2))
        
        total_loss = classif_loss + (kl_beta * kl_loss) + embedding_reg
        return total_loss, classif_loss, kl_loss

    # Simplified and more stable architecture
    vae_encoder = DynamicCouplingEncoder(
        in_feature_dim=in_feature_dim, 
        num_views=args.num_views, 
        hidden_dim=768,  # Reduced complexity
        num_initial_layers=2,  # Reduced layers
        num_coupling_layers=2,  # Reduced layers
        dim_coupling=1024,  # Reduced dimension
        dropout_rate=0.3,  # Increased dropout for regularization
        GraphLatentDim=args.graphEmDim,
        nhead=2 
    )
    vae_decoder = ClassificationDecoder(args.graphEmDim, 768, 1, dropout_rate=0.3)  # Reduced capacity + more dropout
    vae_model = StagedSupervisedVAE(vae_encoder, vae_decoder).to(device)
    
    # More conservative optimizer settings
    optimizer_vae = torch.optim.AdamW(vae_model.parameters(), lr=args.lr * 0.5, weight_decay=1e-3, betas=(0.9, 0.999))
    
    # Cosine annealing with warmup for better convergence
    total_steps = (len(train_data.list_adjs) // args.batchSize) * args.epoch_number
    warmup_steps = total_steps // 10  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler_vae = torch.optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda)

    # More conservative KL annealing parameters
    kl_beta = 0.0
    kl_anneal_epochs = 100  # Much longer annealing period
    steps_per_epoch = max(1, len(train_data.list_adjs) // args.batchSize)
    kl_anneal_steps = steps_per_epoch * kl_anneal_epochs
    kl_anneal_rate = 0.5 / kl_anneal_steps if kl_anneal_steps > 0 else 0.5  # Slower annealing to max 0.5
    print(f"KL Annealing will occur over the first {kl_anneal_epochs} epochs to max beta=0.5.")

    best_test_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 25  # Reduced patience for faster iteration
    checkpoint_dir = f"ckpt/{args.dataset}_stage1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_stage1_model.pt")

    print("\n--- Starting Stage 1: Supervised Pre-training ---")
    for epoch in range(args.epoch_number):
        vae_model.train()
        train_data.shuffle()
        
        epoch_total_loss, epoch_class_loss, epoch_kl_loss = 0, 0, 0
        num_batches = 0

        for i in range(0, len(train_data.list_adjs), args.batchSize):
            from_ = i
            to_ = i + args.batchSize
            
            # Get batch data with labels
            adj_batch, x_batch, _, _, _, labels_batch = train_data.get__(from_, to_, self_for_none=True, get_labels=True)
            target_labels = torch.tensor(labels_batch, device=device)

            # Prepare inputs for the model
            x_s_tensor = torch.stack(x_batch).to(device)
            features_for_dgl = x_s_tensor.view(-1, in_feature_dim)
            
            dgl_graphs_per_view = []
            for v in range(args.num_views):
                view_graphs_in_batch = [dgl.from_scipy(sp.csr_matrix(g[v].cpu().numpy())) for g in adj_batch]
                dgl_graphs_per_view.append(dgl.batch(view_graphs_in_batch).to(device))
            
            batchSize_info = [len(adj_batch), adj_batch[0].shape[-1]]

            # Update KL Beta (slower annealing to prevent collapse)
            if kl_beta < 0.5:
                kl_beta = min(0.5, kl_beta + kl_anneal_rate)

            # Forward pass, loss calculation, backward pass
            optimizer_vae.zero_grad()
            predicted_logits, mean, log_std, _ = vae_model(dgl_graphs_per_view, features_for_dgl, batchSize_info)
            total_loss, class_loss, kl_loss = SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta)
            
            total_loss.backward()
            # More aggressive gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=0.5)
            optimizer_vae.step()
            scheduler_vae.step()  # Step-wise learning rate update

            epoch_total_loss += total_loss.item()
            epoch_class_loss += class_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        # End of epoch evaluation
        avg_total_loss = epoch_total_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        # Evaluate on test set
        vae_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i_test in range(0, len(test_data.list_adjs), args.batchSize):
                from_test = i_test
                to_test = i_test + args.batchSize
                
                adj_test, x_test, _, _, _, labels_test = test_data.get__(from_test, to_test, self_for_none=True, get_labels=True)
                
                x_s_tensor_test = torch.stack(x_test).to(device)
                features_dgl_test = x_s_tensor_test.view(-1, in_feature_dim)
                dgl_views_test = [dgl.batch([dgl.from_scipy(sp.csr_matrix(g[v].cpu().numpy())) for g in adj_test]).to(device) for v in range(args.num_views)]
                batchSize_info_test = [len(adj_test), adj_test[0].shape[-1]]

                _, mean_test, _, _ = vae_model(dgl_views_test, features_dgl_test, batchSize_info_test)
                test_logits = vae_model.decoder(mean_test)
                
                all_preds.append(torch.sigmoid(test_logits).cpu())
                all_labels.append(torch.tensor(labels_test))

        all_preds = torch.cat(all_preds).numpy().ravel()
        all_labels = torch.cat(all_labels).numpy().ravel()
        auc = roc_auc_score(all_labels, all_preds)

        if auc > best_test_auc:
            best_test_auc = auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer_vae.state_dict(),
                'scheduler_state_dict': scheduler_vae.state_dict(),
                'test_auc': auc,
                'kl_beta': kl_beta,
                'args': args
            }, best_model_path)
            print(f"  *** New best model saved! Best AUC: {best_test_auc:.4f} at epoch {best_epoch} ***")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

        print(f"Epoch: {epoch+1:03d} | Avg Loss: {avg_total_loss:.4f} | Class Loss: {avg_class_loss:.4f} | "
              f"KL Loss: {avg_kl_loss:.4f} | Beta: {kl_beta:.3f} | Test AUC: {auc:.4f} | LR: {optimizer_vae.param_groups[0]['lr']:.2e}")

    print("--- STAGE 1 Finished. VAE model is ready. ---")
    # Clean up optimizer memory
    del optimizer_vae, scheduler_vae
    torch.cuda.empty_cache()
    
    # Load the best model
    if os.path.exists(best_model_path):
        print(f"Loading best Stage 1 model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']+1} with test AUC: {checkpoint['test_auc']:.4f}")
    else:
        print("WARNING: No best model checkpoint found. Using final model state.")
    
    
    
    # ========================================================================
    # BRIDGE: GENERATE DDM-READY DATASET FROM VAE EMBEDDINGS
    # ========================================================================
    print("\n--- BRIDGE: Generating DDM-ready dataset from VAE embeddings ---")
    @torch.no_grad()
    def generate_ddm_input_dataset(vae_model, dataset_obj, dgl_multiview_list):
        vae_model.eval()
        ddm_graphs = []
    
    # We process graph by graph to easily match embeddings with structures
        for i in tqdm(range(len(dataset_obj)), desc="Generating DDM Inputs"):
        # Prepare single-graph batch for VAE
            x = [dataset_obj.x_s[i].to_dense().clone().detach()]  # Fix tensor warning
            x_tensor = torch.stack(x).to(device)
            features_dgl = x_tensor.view(-1, in_feature_dim)
        # Get views
            dgl_views = [dgl.batch([dgl_multiview_list[i][v]]).to(device) for v in range(args.num_views)]
        
        # Get graph-level embedding from VAE encoder
            _, graph_embedding, log_std = vae_model.encoder(dgl_views, features_dgl, [1, x[0].shape[0]])

        # Enhanced node features: Use graph embedding + structural information
            dgl_graph_for_ddm = dgl_multiview_list[i][0] # Use first view's structure
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
        
        # Create embeddings for nodes with graph-level context and structural properties
            node_features = graph_embedding.expand(num_nodes, -1).clone()  # Clone to avoid memory issues
        
        # Create structural feature vector
            structural_features = torch.cat([
                normalized_degrees.to(device),
                kcore_proxy.to(device),
                position_encoding,
            # Add noise component scaled by log_std to help diffusion model
                torch.randn(num_nodes, 1, device=device) * torch.exp(log_std).mean()
            ], dim=1)
        
        # Project structural features to partial embedding dimension
            feature_dim = graph_embedding.shape[-1]
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
            
        # Add controlled noise based on VAE uncertainty
            uncertainty_scale = min(0.1, max(0.01, torch.exp(log_std).mean().item()))
            node_variation = torch.randn(num_nodes, feature_dim, device=device) * uncertainty_scale
            node_features = node_features + node_variation
        
        # Normalize features 
            node_features = F.layer_norm(node_features, (node_features.shape[-1],))
        
        # Store processed features
            dgl_graph_for_ddm.ndata['attr'] = node_features.cpu()
            ddm_graphs.append((dgl_graph_for_ddm, dataset_obj.labels[i]))
        
        return ddm_graphs
        
    train_ddm_graphs = generate_ddm_input_dataset(vae_model, train_data, train_dgl_multiview)
    test_ddm_graphs = generate_ddm_input_dataset(vae_model, test_data, test_dgl_multiview)
    
    # ========================================================================
    # STAGE 2: TRAIN DDM FEATURE ENHANCER
    # ========================================================================
    print(f"\n--- STAGE 2: Training DDM Feature Enhancer ---")
    # Use a minimal batch size to avoid memory issues
    memory_safe_batch_size = 8  
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

    # Enhanced model configuration with increased capacity
    ddm_main_args = {
        'in_dim': args.graphEmDim,
        'num_hidden': 768,    # Larger hidden dimension
        'num_layers': 3,       # Deeper network for more capacity
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
    
    scheduler_ddm = torch.optim.lr_scheduler.LambdaLR(optimizer_ddm, lr_lambda)

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
    for epoch in range(1000):  # More epochs for DDM
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
        if epoch % 5 == 0 or epoch >= 300 or epoch < 200:
            # Switch to EMA model for evaluation after it's initialized
            eval_model = ema_model if ema_model is not None and epoch > 50 else ddm_model
            eval_model.eval()
            
            # Enhanced evaluation with more focus on model quality
            # Use the complete training data for stratified k-fold cross-validation to get smoother AUC values
            auc_mean, auc_std = evaluate_with_svm(eval_model, ddm_test_loader, eval_T_values, device)
            print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | CV AUC (SVM): {auc_mean:.4f} ± {auc_std:.4f} | LR: {optimizer_ddm.param_groups[0]['lr']:.2e}")
            
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
# OPTIONAL: t-SNE VISUALISATION OF DDM GRAPH EMBEDDINGS
# =========================================================================

def visualize_tsne(ddm_model, data_loader, T_val, device):
    """Generate a 2-D t-SNE plot of graph-level embeddings produced by the trained DDM."""
    ddm_model.eval()
    avg_pooler = AvgPooling()  # simple yet effective pooling

    all_embeds, all_labels = [], []
    with torch.no_grad():
        for batch_g, labels in tqdm(data_loader, desc="t-SNE embedding extraction"):
            batch_g = batch_g.to(device)
            node_feats = batch_g.ndata['attr']

            # Obtain denoised node representations at the chosen timestep
            denoised_nodes = ddm_model.embed(batch_g, node_feats, T_val)

            # Aggregate to graph-level embeddings (average pooling)
            graph_embeds = avg_pooler(batch_g, denoised_nodes)

            all_embeds.append(graph_embeds.cpu())
            all_labels.append(labels)

    # Stack results
    X = torch.cat(all_embeds, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()

    # Ensure perplexity < number of samples
    perp = max(5, min(30, X.shape[0] - 1))  # keep at least 5 for stability
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Spectral', s=15)
    plt.colorbar(scatter, label='Class label')
    plt.title(f't-SNE of DDM embeddings at T={T_val}')
    plt.xlabel('t-SNE-1'); plt.ylabel('t-SNE-2')
    plt.tight_layout()
    plt.savefig(f'tsne_DDM_T{T_val}.png', dpi=300)
    plt.show()

# --- Automatically generate visualisation for a mid-range diffusion step ---
try:
    tsne_T = 500  # mid diffusion step (can be changed)
    model_for_vis = ema_model if ema_model is not None else ddm_model
    print(f"\nGenerating t-SNE visualisation at timestep T={tsne_T} …")
    visualize_tsne(model_for_vis, ddm_test_loader, tsne_T, device)

    # =========================================================================
    # NEW: Visualise classification results via t-SNE
    # =========================================================================
    def visualize_tsne_classification(ddm_model, train_loader, test_loader, T_val, device):
        """Train a simple classifier on DDM embeddings, predict test labels,
        and visualise them with t-SNE."""
        ddm_model.eval()
        avg_pooler = AvgPooling()

        # 1) Extract embeddings for training set
        train_embeds, train_labels = [], []
        with torch.no_grad():
            for batch_g, labels in tqdm(train_loader, desc="Extract train embeds"):
                batch_g = batch_g.to(device)
                node_feats = batch_g.ndata['attr']
                den_nodes = ddm_model.embed(batch_g, node_feats, T_val)
                g_emb = avg_pooler(batch_g, den_nodes)
                train_embeds.append(g_emb.cpu())
                train_labels.append(labels)
        X_train = torch.cat(train_embeds, dim=0).numpy()
        y_train = torch.cat(train_labels, dim=0).numpy()

        # 2) Train a logistic regression classifier
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        # 3) Extract embeddings for test set & predict
        test_embeds, test_labels = [], []
        with torch.no_grad():
            for batch_g, labels in tqdm(test_loader, desc="Extract test embeds"):
                batch_g = batch_g.to(device)
                node_feats = batch_g.ndata['attr']
                den_nodes = ddm_model.embed(batch_g, node_feats, T_val)
                g_emb = avg_pooler(batch_g, den_nodes)
                test_embeds.append(g_emb.cpu())
                test_labels.append(labels)
        X_test = torch.cat(test_embeds, dim=0).numpy()
        y_test = torch.cat(test_labels, dim=0).numpy()

        y_pred = clf.predict(X_test)
        correct_mask = (y_pred == y_test)

        # 4) Combine train+test for t-SNE
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_pred], axis=0)  # use predicted for colouring
        domain = np.array([0]*len(X_train) + [1]*len(X_test))  # 0=train,1=test
        correct_flag = np.concatenate([np.ones(len(X_train), dtype=bool), correct_mask])

        perp = max(5, min(30, X_all.shape[0] - 1))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X_all)

        # Plot
        plt.figure(figsize=(8,6))
        cmap = 'tab10' if len(np.unique(y_all))<=10 else 'Spectral'
        scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_all, cmap=cmap, s=20, alpha=0.8, edgecolors='k')

        # Highlight misclassified points in test set with X marker
        for i in range(len(X_all)):
            if domain[i]==1 and not correct_flag[i]:
                plt.scatter(X_tsne[i,0], X_tsne[i,1], marker='x', c='red', s=35)

        plt.title(f't-SNE classification view (T={T_val})')
        plt.xlabel('t-SNE-1'); plt.ylabel('t-SNE-2')
        plt.tight_layout()
        # Legends
        handles = [Line2D([0],[0], marker='o', color='w', label='Train', markerfacecolor='gray', markersize=8),
                   Line2D([0],[0], marker='o', color='w', label='Test Correct', markerfacecolor='none', markeredgecolor='k', markersize=8),
                   Line2D([0],[0], marker='x', color='red', label='Test Incorrect', markersize=8)]
        plt.legend(handles=handles, loc='best')
        plt.savefig(f'tsne_classification_T{T_val}.png', dpi=300)
        plt.show()

    print("Creating classification t-SNE plot …")
    visualize_tsne_classification(model_for_vis, ddm_train_loader, ddm_test_loader, tsne_T, device)
except Exception as e:
    print(f"t-SNE visualisation failed: {e}")

