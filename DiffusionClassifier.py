# In main.py or a new file

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Diffusion Utilities ---
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_diffusion_variables(timesteps):
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance
    }

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- The Correct Model for This Task ---
class DiffusionClassifier(nn.Module):
    def __init__(self, num_classes, graph_embedding_dim, time_embedding_dim=128):
        super().__init__()
        # ... (This is the same simple MLP-based model from before)
        self.num_classes = num_classes
        self.graph_embedding_dim = graph_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        self.model = nn.Sequential(
            nn.Linear(num_classes + graph_embedding_dim + time_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, noisy_label, graph_embedding, t):
        t_embedded = self.time_mlp(t.float().unsqueeze(-1))
        x = torch.cat([noisy_label, graph_embedding, t_embedded], dim=-1)
        return self.model(x)