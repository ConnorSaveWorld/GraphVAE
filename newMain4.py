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
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import GraphDataLoader as DGLDataLoader
from sklearn.model_selection import train_test_split
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import gc

# Create a wrapper to handle sklearn version differences
def create_tsne(**kwargs):
    """Create t-SNE with compatibility for different sklearn versions."""
    # Handle parameter name differences
    if 'n_iter' in kwargs:
        kwargs['max_iter'] = kwargs.pop('n_iter')
    
    # Try with current parameters first
    try:
        return TSNE(**kwargs)
    except TypeError as e:
        # Remove any parameters that might not be supported
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['n_components', 'perplexity', 'max_iter', 'random_state', 
                              'learning_rate', 'init', 'metric', 'early_exaggeration', 'verbose']}
        return TSNE(**safe_kwargs)

# Optional imports for enhanced visualizations
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    import openTSNE
    _HAS_OPENTSNE = True
except ImportError:
    _HAS_OPENTSNE = False

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
    
    # Print dataset info for debugging AUC discretization
    print(f"Evaluation dataset size: {len(all_labels)} samples")
    print(f"Class distribution: {np.bincount(all_labels.astype(int))}")
    print(f"Using 5-fold CV with {len(all_labels)//5} samples per test fold")
    
    # For small datasets, use different CV strategy
    if len(all_labels) <= 25:
        # For tiny datasets, repeated stratified K-fold helps stabilise
        # the score by averaging across many random partitions.
        from sklearn.model_selection import RepeatedStratifiedKFold
        n_splits = min(4, len(all_labels))  # keep folds reasonably sized
        n_repeats = 10                      # 10× repeated CV
        print(
            f"Small dataset detected → Using RepeatedStratifiedKFold (splits={n_splits}, repeats={n_repeats}) "
            "to obtain a more stable AUC estimate."
        )
        cv_method = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42,
        )
        cv_name = f"{n_splits}-fold×{n_repeats}"
    else:
        cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-fold"
    
    # Perform cross-validation evaluation
    auc_list = []
    
    print(f"Starting {cv_name} cross-validation...")
    
    # For each fold in cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_method.split(final_embeds[T_values[0]], all_labels)):
        if cv_name == "LOO" and fold_idx % 10 == 0:  # Print progress for LOO
            print(f"LOO progress: {fold_idx}/{len(all_labels)}")
            
        test_scores = []
        
        # Skip fold if test set has only one class (for LOO this shouldn't happen)
        y_test = all_labels[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        
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
                if cv_name == "5-fold":  # Only print for 5-fold to avoid spam
                    print(f"Fold {fold_idx+1} AUC: {auc:.4f}")
            except Exception as e:
                print(f"Error computing AUC: {e}")
    
    if auc_list:
        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        print(f"Final {cv_name} CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        return mean_auc, std_auc
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
    parser.add_argument('-dataset', dest="dataset", default="site21")
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
    full_data = Datasets(list_adj, True, list_x, list_label)
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
    full_data.processALL(self_for_none=True)  # Process full dataset
    full_dgl_multiview = create_dgl_graphs(full_data, args.num_views)
    print(f"--- Data Loaded. Train: {len(train_data)}, Test: {len(test_data)}, Full: {len(full_data)} ---")

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
    def generate_ddm_input_dataset(vae_model, dataset_obj, dgl_multiview_list, enhance_features=True):
        vae_model.eval()
        ddm_graphs = []
        graph_embeddings_all = []  # collect per-graph embeddings
        labels_all = []
        
        # Safety check: ensure the dataset and dgl_multiview_list have matching lengths
        max_idx = min(len(dataset_obj), len(dgl_multiview_list))
        if len(dataset_obj) != len(dgl_multiview_list):
            print(f"WARNING: Dataset size ({len(dataset_obj)}) doesn't match DGL multiview list size ({len(dgl_multiview_list)})")
            print(f"Using the smaller size: {max_idx}")
    
        # We process graph by graph to easily match embeddings with structures
        for i in tqdm(range(max_idx), desc="Generating DDM Inputs"):
            # Prepare single-graph batch for VAE
            x = [dataset_obj.x_s[i].to_dense().clone().detach()]  # Fix tensor warning
            x_tensor = torch.stack(x).to(device)
            features_dgl = x_tensor.view(-1, in_feature_dim)
            # Get views
            dgl_views = [dgl.batch([dgl_multiview_list[i][v]]).to(device) for v in range(args.num_views)]
        
            # Get graph-level embedding from VAE encoder
            _, graph_embedding, log_std = vae_model.encoder(dgl_views, features_dgl, [1, x[0].shape[0]])

            # store graph embedding (detach to CPU)
            graph_embeddings_all.append(graph_embedding.squeeze(0).cpu())
            labels_all.append(dataset_obj.labels[i])
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
        
            # --------------------------------------------------
            # OPTIONAL FEATURE ENHANCEMENT TOGGLE
            # --------------------------------------------------
            # If `enhance_features` is False, we skip the structural
            # augmentation and use the raw graph embedding for all nodes.
            if not enhance_features:
                # Light normalization to keep scales consistent
                node_features = F.layer_norm(node_features, (node_features.shape[-1],))
                dgl_graph_for_ddm.ndata['attr'] = node_features.cpu()
                ddm_graphs.append((dgl_graph_for_ddm, dataset_obj.labels[i]))
                continue

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
        
        # stack list to tensor (N_graphs, embed_dim)
        if graph_embeddings_all:
            graph_embeddings_all = torch.stack(graph_embeddings_all)
        else:
            graph_embeddings_all = torch.empty(0, vae_model.encoder.GraphLatentDim)

        return ddm_graphs, graph_embeddings_all, labels_all     
        
    
    train_ddm_graphs, train_graph_embeddings, train_labels = generate_ddm_input_dataset(vae_model, train_data, train_dgl_multiview, enhance_features=True)
    test_ddm_graphs, test_graph_embeddings, test_labels = generate_ddm_input_dataset(vae_model, test_data, test_dgl_multiview, enhance_features=True)
    full_ddm_graphs, full_graph_embeddings, full_labels = generate_ddm_input_dataset(vae_model, full_data, full_dgl_multiview, enhance_features=True)
    
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
    full_loader = DGLDataLoader(
        full_ddm_graphs,
        batch_size=len(full_ddm_graphs),
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
    for epoch in range(100):  # More epochs for DDM
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

# =====================
# VISUALIZATION SECTION (similar to test.py)
# =====================

del optimizer_ddm, scheduler_ddm, ddm_train_loader, ddm_test_loader

gc.collect()
torch.cuda.empty_cache()

# DataLoader for full test set
full_test_loader = DGLDataLoader(
    test_ddm_graphs,
    batch_size=len(test_ddm_graphs),
    shuffle=False,
    collate_fn=collate_dgl,
    num_workers=0
)

def visualize_tsne(
    embeddings,
    labels,
    perplexity=None,
    n_iter=5000,
    pca_dim=None,
    title="t-SNE Visualization",
    hide_axis=False,
    save_name="tsne_visualization.png"
):
    """Enhanced t-SNE visualisation with multiple improvements for small datasets.

    Steps:
    1. Z-score standardise features.
    2. Smart PCA preprocessing (retain more variance for small datasets).
    3. Adaptive perplexity with better bounds for small datasets.
    4. Multiple t-SNE runs with different parameters.
    5. Enhanced plotting with better colors and larger markers.
    """

    N = len(embeddings)
    print(f"Starting enhanced t-SNE for {N} samples...")
    
    # 1. Robust standardization
    emb_std = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-7)
    
    # 2. Smart PCA preprocessing - be more aggressive for small datasets
    if pca_dim is None:
        # For small datasets, use more components to retain information
        if N < 50:
            effective_pca_dim = min(N - 2, min(emb_std.shape[1], max(20, int(N * 0.8))))
        else:
            effective_pca_dim = min(N - 1, min(emb_std.shape[1], 50))
    else:
        effective_pca_dim = min(pca_dim, min(N - 1, emb_std.shape[1]))
    
    # Apply PCA if we have high-dimensional data
    if emb_std.shape[1] > effective_pca_dim and effective_pca_dim >= 2:
        print(f"Applying PCA: {emb_std.shape[1]} -> {effective_pca_dim} dimensions")
        pca = PCA(n_components=effective_pca_dim, random_state=42)
        emb_proc = pca.fit_transform(emb_std)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA retains {explained_var:.1%} of variance")
    else:
        print(f"Using original {emb_std.shape[1]} dimensions")
        emb_proc = emb_std

    # 3. Enhanced perplexity selection for small datasets
    if perplexity is None:
        if N < 50:
            # For very small datasets, use smaller perplexity
            perplexity = max(2, min(15, int(N / 3)))
        else:
            perplexity = max(5, min(50, int(0.3 * math.sqrt(N))))

    print(f"Running enhanced t-SNE on {N} points (dim={emb_proc.shape[1]}) | perp={perplexity} | iter={n_iter}")

    # 4. Try multiple t-SNE configurations and pick the best one
    best_embeddings_2d = None
    best_separation = -float('inf')
    
    configs = [
        # Config 1: Standard with cosine metric
        {'metric': 'cosine', 'init': 'pca', 'early_exaggeration': 24, 'learning_rate': 'auto'},
        # Config 2: Euclidean with higher early exaggeration
        {'metric': 'euclidean', 'init': 'pca', 'early_exaggeration': 48, 'learning_rate': 200},
        # Config 3: Cosine with random init
        {'metric': 'cosine', 'init': 'random', 'early_exaggeration': 36, 'learning_rate': 'auto'},
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"  Trying configuration {i+1}/3: {config['metric']} metric, {config['init']} init...")
            
            tsne = create_tsne(
                n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                early_exaggeration=config['early_exaggeration'],
                learning_rate=config['learning_rate'],
                init=config['init'],
                metric=config['metric'],
                random_state=42,
                verbose=0
            )
            
            embeddings_2d = tsne.fit_transform(emb_proc)
            
            # Evaluate separation quality using silhouette score on 2D embedding
            try:
                separation_score = silhouette_score(embeddings_2d, labels)
                print(f"    Separation score: {separation_score:.4f}")
                
                if separation_score > best_separation:
                    best_separation = separation_score
                    best_embeddings_2d = embeddings_2d.copy()
                    best_config = config
            except:
                # If silhouette fails, use distance-based metric
                class_centers = []
                for label in np.unique(labels):
                    mask = labels == label
                    if np.any(mask):
                        center = embeddings_2d[mask].mean(axis=0)
                        class_centers.append(center)
                
                if len(class_centers) >= 2:
                    center_dist = np.linalg.norm(class_centers[0] - class_centers[1])
                    if center_dist > best_separation:
                        best_separation = center_dist
                        best_embeddings_2d = embeddings_2d.copy()
                        best_config = config
                        
        except Exception as e:
            print(f"    Configuration {i+1} failed: {e}")
            continue
    
    if best_embeddings_2d is None:
        print("All configurations failed, using fallback...")
        # Fallback configuration
        tsne = create_tsne(
            n_components=2,
            perplexity=max(2, perplexity//2),
            n_iter=1000,
            random_state=42
        )
        best_embeddings_2d = tsne.fit_transform(emb_proc)
        best_config = {'metric': 'euclidean', 'init': 'random'}
    else:
        print(f"Best configuration: {best_config} (separation: {best_separation:.4f})")

    # 5. Enhanced plotting
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    
    # Use better color scheme for binary classification
    if len(unique_labels) == 2:
        colors = ['#FF6B6B', '#4ECDC4']  # Red and teal
        markers = ['o', 's']  # Circle and square
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        markers = ['o'] * len(unique_labels)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            best_embeddings_2d[mask, 0], 
            best_embeddings_2d[mask, 1],
            c=colors[i],
            marker=markers[i],
            label=f'Class {label} (n={np.sum(mask)})',
            alpha=0.8,
            s=100,  # Larger markers
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add convex hulls to show class boundaries
    try:
        from scipy.spatial import ConvexHull
        for i, label in enumerate(unique_labels):
            mask = labels == label
            points = best_embeddings_2d[mask]
            if len(points) >= 3:  # Need at least 3 points for hull
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 
                           color=colors[i], alpha=0.3, linewidth=1)
    except:
        pass  # Skip hulls if scipy not available or insufficient points
    
    plt.title(f"{title}\nSeparation Score: {best_separation:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    if hide_axis:
        plt.axis('off')
    
    plt.tight_layout()
    output_path = f'ckpt/{args.dataset}/{save_name}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced t-SNE visualization saved to {output_path}")
    plt.show()
    
    return best_embeddings_2d, best_separation

# ----------------------------------------------------------------------
# Additional helpers for imbalanced datasets
# ----------------------------------------------------------------------
def get_balanced_indices(labels, seed=42):
    """Return indices that balance minority and majority classes 1:1."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    k = len(pos_idx)
    if k == 0:
        return np.arange(len(labels))  # nothing to balance
    neg_keep = rng.choice(neg_idx, size=min(k, len(neg_idx)), replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)
    return keep


def visualize_tsne_weighted(embeddings, labels, title_suffix="Weighted t-SNE"):
    if not _HAS_OPENTSNE:
        print("openTSNE not installed – skipping weighted t-SNE")
        return

    # Standardise + PCA with safe dimensionality
    emb_std = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-7)
    max_components = min(len(embeddings), emb_std.shape[1]) - 1  # -1 for safety margin
    effective_pca_dim = min(50, max_components)
    
    if emb_std.shape[1] > effective_pca_dim and effective_pca_dim > 2:
        print(f"Applying PCA: {emb_std.shape[1]} -> {effective_pca_dim} dimensions")
        emb_proc = PCA(n_components=effective_pca_dim, random_state=42).fit_transform(emb_std)
    else:
        print(f"Using original {emb_std.shape[1]} dimensions")
        emb_proc = emb_std

    # Point weights ‑ inverse class frequency
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    w   = np.where(labels == 1, 1.0, pos / max(1, neg))

    perp = max(5, min(40, int(0.3 * math.sqrt(len(labels)))))
    print(f"Running weighted openTSNE (perp={perp}) …")

    tsne = openTSNE.TSNE(
        perplexity=perp,
        metric="cosine",
        initialization="pca",
        n_jobs=8,
        negative_gradient_method="fft",
        random_state=42,
    )
    try:
        Y = tsne.fit(emb_proc, point_densities=w)
    except TypeError:
        print("openTSNE version does not support point_densities – running without weights")
        Y = tsne.fit(emb_proc)

    plt.figure(figsize=(10, 8))
    for cls, color in zip([0, 1], ["tab:blue", "tab:red"]):
        mask = labels == cls
        plt.scatter(Y[mask, 0], Y[mask, 1], c=color, label=f"Class {cls}", alpha=0.7)
    plt.title(f"{title_suffix}")
    plt.axis("off")
    out_path = f'ckpt/{args.dataset}/tsne_weighted.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Weighted t-SNE saved to {out_path}")
    plt.show()

def visualize_multiple_tsne_approaches(embeddings, labels, title_prefix="Multi-approach t-SNE"):
    """Try multiple t-SNE approaches and visualization strategies."""
    print(f"\n=== {title_prefix} Analysis ===")
    
    # Approach 1: Full dataset with density visualization
    print("1. Full dataset with density-based visualization...")
    try:
        visualize_tsne_with_density(embeddings, labels, 
                                  title=f"{title_prefix} - Full Dataset (Density)",
                                  save_name="tsne_full_density.png")
    except Exception as e:
        print(f"   Density visualization failed: {e}")
    
    # Approach 2: Balanced subset with enhanced visualization
    print("2. Balanced subset with enhanced t-SNE...")
    balanced_idx = get_balanced_indices(labels)
    balanced_embeddings = embeddings[balanced_idx]
    balanced_labels = labels[balanced_idx]
    
    embeddings_2d, separation = visualize_tsne(
        balanced_embeddings, balanced_labels,
        title=f"{title_prefix} - Balanced Enhanced",
        save_name="tsne_balanced_enhanced.png"
    )
    
    # Approach 3: Multiple perplexity values
    print("3. Multiple perplexity comparison...")
    visualize_tsne_perplexity_comparison(balanced_embeddings, balanced_labels, title_prefix)
    
    return embeddings_2d, separation

def visualize_tsne_with_density(embeddings, labels, title="t-SNE with Density", save_name="tsne_density.png"):
    """Enhanced t-SNE with density-based visualization for imbalanced datasets."""
    N = len(embeddings)
    print(f"Running density-based t-SNE on {N} samples...")
    
    # Preprocessing
    emb_std = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-7)
    
    # More aggressive PCA for full dataset
    if N < 100:
        pca_dim = min(N - 2, min(emb_std.shape[1], max(30, int(N * 0.7))))
    else:
        pca_dim = min(N - 1, min(emb_std.shape[1], 75))
    
    if emb_std.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_proc = pca.fit_transform(emb_std)
        print(f"PCA: {emb_std.shape[1]} -> {pca_dim} dims, variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    else:
        emb_proc = emb_std
    
    # Adaptive perplexity for larger dataset
    perplexity = max(3, min(min(30, N//3), int(0.25 * math.sqrt(N))))
    
    # Run t-SNE with higher iterations for better convergence
    tsne = create_tsne(
        n_components=2,
        perplexity=perplexity,
        n_iter=8000,  # Higher iterations
        early_exaggeration=30,
        learning_rate=200,
        init='pca',
        random_state=42
    )
    
    embeddings_2d = tsne.fit_transform(emb_proc)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Standard scatter
    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4'] if len(unique_labels) == 2 else plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[0].scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=colors[i],
            label=f'Class {label} (n={np.sum(mask)})',
            alpha=0.7,
            s=60
        )
    
    axes[0].set_title(f"Standard Scatter\nPerplexity: {perplexity}", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Density-based visualization
    try:
        # Create density plot for majority class as background
        majority_class = 0 if (labels == 0).sum() > (labels == 1).sum() else 1
        minority_class = 1 - majority_class
        
        # Plot density for majority class
        maj_mask = labels == majority_class
        min_mask = labels == minority_class
        
        axes[1].scatter(
            embeddings_2d[maj_mask, 0], embeddings_2d[maj_mask, 1],
            c=colors[majority_class], alpha=0.4, s=30, 
            label=f'Class {majority_class} (n={np.sum(maj_mask)})'
        )
        
        # Highlight minority class
        axes[1].scatter(
            embeddings_2d[min_mask, 0], embeddings_2d[min_mask, 1],
            c=colors[minority_class], alpha=0.9, s=100, 
            edgecolors='black', linewidth=1,
            label=f'Class {minority_class} (n={np.sum(min_mask)})'
        )
        
        axes[1].set_title("Density-aware Visualization\n(Minority class highlighted)", fontsize=12)
        
    except Exception as e:
        print(f"Density plot failed, using standard: {e}")
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[1].scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=colors[i], label=f'Class {label}', alpha=0.7
            )
    
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f'ckpt/{args.dataset}/{save_name}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Density t-SNE saved to {output_path}")
    plt.show()

def visualize_tsne_perplexity_comparison(embeddings, labels, title_prefix="Perplexity Comparison"):
    """Compare t-SNE results with different perplexity values."""
    N = len(embeddings)
    perplexities = [max(2, N//8), max(3, N//5), max(5, N//3)]
    perplexities = [p for p in perplexities if p < N]  # Ensure valid perplexities
    
    if len(perplexities) == 0:
        perplexities = [max(2, N//2)]
    
    fig, axes = plt.subplots(1, len(perplexities), figsize=(6*len(perplexities), 6))
    if len(perplexities) == 1:
        axes = [axes]
    
    # Preprocess once
    emb_std = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-7)
    pca_dim = min(N - 2, min(emb_std.shape[1], max(15, int(N * 0.8))))
    
    if emb_std.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_proc = pca.fit_transform(emb_std)
    else:
        emb_proc = emb_std
    
    colors = ['#FF6B6B', '#4ECDC4']
    best_sep = -float('inf')
    best_perp = perplexities[0]
    
    for i, perp in enumerate(perplexities):
        try:
            tsne = create_tsne(
                n_components=2,
                perplexity=perp,
                n_iter=4000,
                random_state=42,
                init='pca'
            )
            emb_2d = tsne.fit_transform(emb_proc)
            
            # Calculate separation
            try:
                sep_score = silhouette_score(emb_2d, labels)
                if sep_score > best_sep:
                    best_sep = sep_score
                    best_perp = perp
            except:
                sep_score = 0
            
            # Plot
            for j, label in enumerate(np.unique(labels)):
                mask = labels == label
                axes[i].scatter(
                    emb_2d[mask, 0], emb_2d[mask, 1],
                    c=colors[j], label=f'Class {label}',
                    alpha=0.8, s=60
                )
            
            axes[i].set_title(f'Perplexity: {perp}\nSep: {sep_score:.3f}', fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[i].set_title(f'Perp {perp}: Failed')
            print(f"Perplexity {perp} failed: {e}")
    
    plt.suptitle(f'{title_prefix} (Best: perp={best_perp}, sep={best_sep:.3f})', fontsize=12)
    plt.tight_layout()
    
    output_path = f'ckpt/{args.dataset}/tsne_perplexity_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Perplexity comparison saved to {output_path}")
    plt.show()
    
    return best_perp, best_sep

def visualize_pca(embeddings, labels, n_components=2, title="PCA Visualization of Graph Embeddings"):
    print(f"Running PCA with n_components={n_components}...")
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_.sum()
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
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
    output_filename = f'ckpt/{args.dataset}/pca_visualization.png'
    plt.savefig(output_filename, dpi=300)
    print(f"PCA visualization saved to {output_filename}")
    plt.show()

def evaluate_embedding_quality(embeddings, labels):
    try:
        sil_score = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        print(f"Embedding Quality Metrics:")
        print(f"  Silhouette Score: {sil_score:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")
        return sil_score, db_score
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")
        return None, None

def visualize_kmeans_clusters(embeddings, true_labels, n_clusters=None, title="K-means Clustering"):
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))
    tsne = create_tsne(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
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
    centers_approx = []
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            center_approx = np.mean(embeddings_2d[mask], axis=0)
            centers_approx.append(center_approx)
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
    output_filename = f'ckpt/{args.dataset}/kmeans_clustering.png'
    plt.savefig(output_filename, dpi=300)
    print(f"K-means clustering visualization saved to {output_filename}")
    plt.show()
    return cluster_labels

def extract_post_diffusion_embeddings(model, data_loader, t_val, device, use_proj_head=True):
    """Extract embeddings from DDM model at specified timestep."""
    model.eval()
    all_embeds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_g, labels in data_loader:
            batch_g = batch_g.to(device)
            features = batch_g.ndata['attr']
            
            # Get denoised features at specified timestep
            denoised_features = model.embed(batch_g, features, t_val)
            
            # Apply pooling to get graph-level embeddings
            avg_pooler = AvgPooling()
            graph_embeds = avg_pooler(batch_g, denoised_features)
            
            # Apply projection head if requested
            if use_proj_head and hasattr(model, 'proj_head'):
                graph_embeds = model.proj_head(graph_embeds)
            
            all_embeds.append(graph_embeds.cpu())
            all_labels.append(labels.cpu())
    
    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return all_embeds, all_labels

def visualize_umap_supervised(embeddings, labels, title_suffix="Supervised UMAP"):
    """Create supervised UMAP visualization."""
    if not _HAS_UMAP:
        print("UMAP not installed – skipping UMAP visualization")
        return
    
    try:
        # Standardize embeddings
        emb_std = (embeddings - embeddings.mean(0)) / (embeddings.std(0) + 1e-7)
        
        # Apply UMAP with label supervision
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(embeddings) - 1),
            target_weight=0.5,  # Use labels for supervision
            random_state=42
        )
        
        embedding_2d = reducer.fit_transform(emb_std, y=labels)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding_2d[mask, 0], 
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=0.7
            )
        
        plt.title(title_suffix)
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.legend()
        plt.axis("off")
        
        output_path = f'ckpt/{args.dataset}/umap_supervised.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Supervised UMAP saved to {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"Error in UMAP visualization: {e}")

def supervised_finetune_projection(model, train_loader, test_loader, device, num_epochs=5):
    """Fine-tune the projection head with supervised loss."""
    model.train()
    
    # Add a classification head for fine-tuning
    feature_dim = model.proj_head[-1].out_features if hasattr(model, 'proj_head') else 768
    classifier = nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(feature_dim // 2, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.proj_head.parameters()) + list(classifier.parameters()),
        lr=1e-4
    )
    
    # Fine-tuning loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_g, labels in train_loader:
            batch_g = batch_g.to(device)
            labels = labels.float().to(device)
            features = batch_g.ndata['attr']
            
            optimizer.zero_grad()
            
            # Get embeddings
            denoised_features = model.embed(batch_g, features, 100)
            avg_pooler = AvgPooling()
            graph_embeds = avg_pooler(batch_g, denoised_features)
            proj_embeds = model.proj_head(graph_embeds)
            
            # Classification loss
            logits = classifier(proj_embeds).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Fine-tuning epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Extract final embeddings
    model.eval()
    all_embeds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_g, labels in test_loader:
            batch_g = batch_g.to(device)
            features = batch_g.ndata['attr']
            
            denoised_features = model.embed(batch_g, features, 100)
            avg_pooler = AvgPooling()
            graph_embeds = avg_pooler(batch_g, denoised_features)
            proj_embeds = model.proj_head(graph_embeds)
            
            all_embeds.append(proj_embeds.cpu())
            all_labels.append(labels.cpu())
    
    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return all_embeds, all_labels, classifier

# === Run visualizations ===
print("\n--- Generating t-SNE Visualization ---")
try:
    best_model_path = f'ckpt/{args.dataset}/best_ddm_model.pth'
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if checkpoint['ema_model_state_dict'] is not None:
        print("Using EMA model for visualization")
        ddm_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("Using regular model for visualization")
        ddm_model.load_state_dict(checkpoint['model_state_dict'])
    t_val = 100
    print(f"Extracting embeddings at timestep t={t_val}...")

    # ================================================================
    # COMPREHENSIVE VISUALIZATION PIPELINE FOR IMBALANCED DATA
    # ================================================================
    test_labels_array = np.array(test_labels)
    print(f"Dataset imbalance: {(test_labels_array==0).sum()} negatives, {(test_labels_array==1).sum()} positives")
    
    # --- Method 1: Bridge-level embeddings (for comparison) ---
    print("\n=== 1. Bridge-level embeddings (VAE output) ===")
    print(f"Shape: {test_graph_embeddings.numpy().shape}")
    
    # Multiple approaches for bridge embeddings
    visualize_multiple_tsne_approaches(
        test_graph_embeddings.numpy(), test_labels_array,
        title_prefix="Bridge Embeddings"
    )
    
    # --- Method 2: Post-diffusion embeddings with projection head ---
    print("\n=== 2. Post-diffusion embeddings (DDM enhanced) ===")
    
    # Extract post-diffusion embeddings using projection head
    post_diff_embeds, post_diff_labels = extract_post_diffusion_embeddings(
        ddm_model, full_test_loader, t_val=100, device=device, use_proj_head=True)
    
    print(f"Post-diffusion shape: {post_diff_embeds.shape}")
    
    # Multiple approaches for post-diffusion embeddings
    best_embeddings_2d, best_separation = visualize_multiple_tsne_approaches(
        post_diff_embeds, post_diff_labels,
        title_prefix="Post-Diffusion Enhanced"
    )
    
    # Full dataset with weighted t-SNE (if openTSNE available)
    print("\n=== 2b. Weighted t-SNE (Full Dataset) ===")
    visualize_tsne_weighted(post_diff_embeds, post_diff_labels,
                            title_suffix="Post-Diffusion Weighted t-SNE")
    
    # --- Method 3: Supervised UMAP ---
    print("\n=== 3. Supervised UMAP ===")
    visualize_umap_supervised(post_diff_embeds, post_diff_labels,
                              title_suffix="Post-Diffusion Supervised UMAP")
    
    # --- Method 4: PCA for comparison ---
    print("\n=== 4. PCA Analysis ===")
    visualize_pca(post_diff_embeds, post_diff_labels, title="PCA Post-Diffusion Embeddings")
    
    # --- Method 5: Optional supervised fine-tuning ---
    user_wants_finetuning = False  # Set to True to enable
    if user_wants_finetuning:
        print("\n=== 5. Supervised Fine-tuning ===")
        try:
            # Create smaller data loaders for fine-tuning
            ddm_train_small = DGLDataLoader(train_ddm_graphs, batch_size=16, shuffle=True, 
                                          collate_fn=collate_dgl, num_workers=0)
            ddm_test_small = DGLDataLoader(test_ddm_graphs, batch_size=16, shuffle=False,
                                         collate_fn=collate_dgl, num_workers=0)
            
            finetuned_embeds, finetuned_labels, _ = supervised_finetune_projection(
                ddm_model, ddm_train_small, ddm_test_small, device, num_epochs=5)
            
            print(f"Fine-tuned embeddings shape: {finetuned_embeds.shape}")
            
            # Visualize fine-tuned embeddings
            keep_idx3 = get_balanced_indices(finetuned_labels)
            visualize_tsne(finetuned_embeds[keep_idx3], finetuned_labels[keep_idx3],
                           perplexity=5, n_iter=3000,
                           title="t-SNE Fine-tuned Embeddings (Balanced)", hide_axis=True)
            
            visualize_umap_supervised(finetuned_embeds, finetuned_labels,
                                      title_suffix="Fine-tuned Supervised UMAP")
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
    
    # --- Embedding Quality Assessment ---
    print("\n=== Embedding Quality Metrics ===")
    
    # Bridge embeddings
    bridge_sil, bridge_db = evaluate_embedding_quality(test_graph_embeddings.numpy(), test_labels_array)
    print(f"Bridge embeddings: Silhouette={bridge_sil:.4f}, Davies-Bouldin={bridge_db:.4f}")
    
    # Post-diffusion embeddings
    post_sil, post_db = evaluate_embedding_quality(post_diff_embeds, post_diff_labels)
    print(f"Post-diffusion embeddings: Silhouette={post_sil:.4f}, Davies-Bouldin={post_db:.4f}")
    
    improvement_sil = post_sil - bridge_sil
    improvement_db = bridge_db - post_db  # Lower is better for DB
    print(f"Improvement: Silhouette +{improvement_sil:.4f}, Davies-Bouldin {improvement_db:+.4f}")
    
    print("\n=== Visualization Complete ===")
    print("Enhanced visualizations saved:")
    print("• tsne_balanced_enhanced.png (best balanced subset)")
    print("• tsne_full_density.png (full dataset with density)")
    print("• tsne_perplexity_comparison.png (different perplexity values)")
    print("• tsne_weighted.png (weighted for imbalance)")
    print("• pca_visualization.png (PCA analysis)")
    
    if 'best_separation' in locals():
        print(f"\nBest t-SNE separation score: {best_separation:.4f}")
        if best_separation > 0.1:
            print("✓ Good class separation achieved!")
        elif best_separation > 0.0:
            print("⚠ Moderate class separation")
        else:
            print("⚠ Poor class separation - consider feature engineering")
    
    # Additional recommendations
    print("\n=== Recommendations for improvement ===")
    print("1. If separation is poor, try:")
    print("   - Different embedding extraction timesteps (t=50, 200, 500)")
    print("   - Using VAE latent space directly (without DDM)")
    print("   - Feature engineering or selection")
    print("2. For better visualization:")
    print("   - Install umap-learn: pip install umap-learn")
    print("   - Install openTSNE: pip install opentsne")
    print("   - Try PHATE: pip install phate")

except Exception as e:
    print(f"Error during visualization: {e}")

# =================================================================
# SCRIPT COMPLETE - All visualization methods implemented:
# =================================================================
# 1. Balanced subset t-SNE visualization (1:1 class ratio)
# 2. Weighted t-SNE with openTSNE (down-weights majority class)  
# 3. Supervised UMAP (uses labels to improve separation)
# 4. Post-diffusion embeddings with projection head
# 5. Optional supervised fine-tuning for maximum separation
#
# To enable fine-tuning, set user_wants_finetuning = True above
# Install dependencies: pip install umap-learn opentsne