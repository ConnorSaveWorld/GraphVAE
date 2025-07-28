#
# newMain.py (VAE Pre-training + DDM Feature Enhancement)
#
import logging
import argparse
import os
import random
import warnings

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

class Denoising_Unet(nn.Module):
    """ The GATv2-based U-Net for the DDM. """
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, nhead, activation, feat_drop, attn_drop, negative_slope, norm):
        super(Denoising_Unet, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.norm = norm
        if norm: self.norm_layers = nn.ModuleList()

        # Input to the first GAT layer is concatenated features + time embedding
        self.gat_layers.append(dgl.nn.GATv2Conv(in_dim + num_hidden, num_hidden // nhead, nhead, feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        if norm: self.norm_layers.append(nn.LayerNorm(num_hidden))

        for _ in range(1, num_layers):
            self.gat_layers.append(dgl.nn.GATv2Conv(num_hidden, num_hidden // nhead, nhead, feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
            if norm: self.norm_layers.append(nn.LayerNorm(num_hidden))

        self.output = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, out_dim))

    def forward(self, g, x_t, time_embed):
        # Concatenate time embedding with node features
        h = torch.cat([x_t, time_embed], dim=1)

        for l, layer in enumerate(self.gat_layers):
            h_in = h
            h = layer(g, h).flatten(1)
            # Simple residual connection
            if h.shape == h_in.shape:
                h = h + h_in
            if self.norm:
                h = self.norm_layers[l](h)

        return self.output(h), h

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
    """ The Denoising Diffusion Model for unsupervised feature enhancement. """
    def __init__(self, in_dim, num_hidden, num_layers, nhead, activation, feat_drop, attn_drop, norm, **kwargs):
        super(DDM, self).__init__()
        self.T = kwargs.get('T', 1000)
        self.alpha_l = kwargs.get('alpha_l', 2.0)
        beta_schedule = kwargs.get('beta_schedule', 'linear')
        beta_1 = kwargs.get('beta_1', 1e-4)
        beta_T = kwargs.get('beta_T', 0.02)

        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, self.T)
        self.register_buffer('betas', beta)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.net = Denoising_Unet(in_dim, num_hidden, in_dim, num_layers, nhead, activation, feat_drop, attn_drop, 0.2, norm)
        self.time_embedding = nn.Embedding(self.T, num_hidden)

    def loss_fn(self, x, y):
        mse_loss = F.mse_loss(x, y)
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        cos_loss = (1 - (x_norm * y_norm).sum(dim=-1)).pow_(self.alpha_l).mean()
        return mse_loss + 0.1 * cos_loss

    def sample_q(self, t, x, batch_num_nodes=None):
        miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        with torch.no_grad():
            noise = F.layer_norm(noise, (noise.shape[-1],))
        noise = noise * std + miu
        noise = torch.sign(x) * torch.abs(noise)
    
    # Extract coefficients for each graph - shape will be [batch_size]
        sqrt_alphas_bar = torch.gather(self.sqrt_alphas_bar, index=t, dim=0).float()
        sqrt_one_minus_alphas_bar = torch.gather(self.sqrt_one_minus_alphas_bar, index=t, dim=0).float()
    
    # If we have batch_num_nodes, repeat coefficients for each node
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
    
    # Now apply the diffusion process
        x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * noise
    
    # Handle time embeddings
        time_embed = self.time_embedding(t)  # Shape: [batch_size, num_hidden]
    
    # Repeat time embedding for each node
        if batch_num_nodes is not None:
            time_embed = time_embed.repeat_interleave(batch_num_nodes, dim=0)
        else:
            nodes_per_graph = num_nodes // batch_size
            time_embed = time_embed.repeat_interleave(nodes_per_graph, dim=0)
    
        return x_t, time_embed

    def forward(self, g, x):
    # Get batch information
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
    
    # Sample one timestamp per graph (not per node)
        t = torch.randint(self.T, size=(batch_size,), device=x.device)
    
    # Pass batch_num_nodes to properly handle time embeddings
        x_t, time_embed = self.sample_q(t, x, batch_num_nodes)
    
    # No need for broadcasting - time_embed is already at node level
        denoised_x, _ = self.net(g, x_t, time_embed)
        return self.loss_fn(denoised_x, x)

    @torch.no_grad()
    def embed(self, g, x, t_int):
    # Get batch information
        batch_num_nodes = g.batch_num_nodes()
        batch_size = len(batch_num_nodes)
    
    # Create timestamp tensor for each graph
        t = torch.full((batch_size,), t_int, device=x.device, dtype=torch.long)
    
    # Pass batch_num_nodes to properly handle time embeddings
        x_t, time_embed = self.sample_q(t, x, batch_num_nodes)
    
    # No need for broadcasting - time_embed is already at node level
        denoised_x, _ = self.net(g, x_t, time_embed)
        return denoised_x

# =========================================================================
# STAGE 2: EVALUATION AND HELPER FUNCTIONS
# =========================================================================

def collate_dgl(batch):
    graphs, labels = map(list, zip(*batch))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def evaluate_with_svm(model, data_loader, T_values, device):
    """Evaluates the DDM-enhanced embeddings using an SVM classifier."""
    model.eval()
    pooler = dgl.nn.pytorch.glob.AvgPooling()
    all_labels, embeds_by_T = [], {t: [] for t in T_values}

    for batch_g, labels in data_loader:
        batch_g, labels = batch_g.to(device), labels.to(device)
        # The 'attr' ndata field holds the broadcasted VAE embeddings
        feat = batch_g.ndata['attr']
        all_labels.append(labels.cpu())

        for t_val in T_values:
            # Get DDM-enhanced node features
            denoised_nodes = model.embed(batch_g, feat, t_val)
            # Pool to get graph-level embeddings
            graph_level_embed = pooler(batch_g, denoised_nodes)
            embeds_by_T[t_val].append(graph_level_embed.cpu())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    final_embeds = {t: torch.cat(embeds, dim=0).numpy() for t, embeds in embeds_by_T.items()}

    # Use cross-validation to get a robust AUC score
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_list = []
    for train_idx, test_idx in kf.split(final_embeds[T_values[0]], all_labels):
        test_scores = []
        for t_val in T_values:
            x_train, x_test = final_embeds[t_val][train_idx], final_embeds[t_val][test_idx]
            y_train, y_test = all_labels[train_idx], all_labels[test_idx]
            svc = SVC(probability=True, C=1.0, random_state=42).fit(x_train, y_train)
            test_scores.append(svc.decision_function(x_test))
        # Average the decision scores from SVMs trained on embeddings from different T
        avg_scores = np.mean(test_scores, axis=0)
        auc_list.append(roc_auc_score(y_test, avg_scores))

    return np.mean(auc_list), np.std(auc_list)

# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage Graph Classification: VAE Pre-training + DDM Enhancement')
    parser.add_argument('-e', dest="epoch_number", default=100, type=int)
    parser.add_argument('-lr', dest="lr", default=1e-4, type=float)
    parser.add_argument('-batchSize', dest="batchSize", default=16, type=int)
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-graphEmDim', dest="graphEmDim", default=128, type=int)
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
    # (This part is identical to your first script)
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
    
    def SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta, kl_threshold=0.5):
        classif_loss = F.binary_cross_entropy_with_logits(
            predicted_logits, target_labels.float().view(-1, 1)
        )
        kl_div = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - log_std.exp().pow(2), dim=1)
        kl_loss = torch.mean(torch.clamp(kl_div, min=kl_threshold))  # Free bits
        total_loss = classif_loss + (kl_beta * kl_loss)
        return total_loss, classif_loss, kl_loss

    vae_encoder = DynamicCouplingEncoder(in_feature_dim, args.num_views, 256, 2, 2, 128, 0.2, args.graphEmDim)
    vae_decoder = ClassificationDecoder(args.graphEmDim, 512, 1)
    vae_model = StagedSupervisedVAE(vae_encoder, vae_decoder).to(device)
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.lr, weight_decay=1e-5)

    # KL annealing parameters
    kl_beta = 0.0
    kl_anneal_epochs = 40  # Anneal over first 40 epochs
    steps_per_epoch = max(1, len(train_data.list_adjs) // args.batchSize)
    kl_anneal_steps = steps_per_epoch * kl_anneal_epochs
    kl_anneal_rate = 1.0 / kl_anneal_steps if kl_anneal_steps > 0 else 1.0
    print(f"KL Annealing will occur over the first {kl_anneal_epochs} epochs.")

    best_test_auc = 0.0
    best_epoch = 0
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

            # Update KL Beta (annealing)
            if kl_beta < 1.0:
                kl_beta = min(1.0, kl_beta + kl_anneal_rate)

            # Forward pass, loss calculation, backward pass
            optimizer_vae.zero_grad()
            predicted_logits, mean, log_std, _ = vae_model(dgl_graphs_per_view, features_for_dgl, batchSize_info)
            total_loss, class_loss, kl_loss = SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            optimizer_vae.step()

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer_vae.state_dict(),
                'test_auc': auc,
                'kl_beta': kl_beta,
                'args': args
            }, best_model_path)
            print(f"  *** New best model saved! Best AUC: {best_test_auc:.4f} at epoch {best_epoch} ***")

        print(f"Epoch: {epoch+1:03d} | Avg Loss: {avg_total_loss:.4f} | Class Loss: {avg_class_loss:.4f} | "
              f"KL Loss: {avg_kl_loss:.4f} | Beta: {kl_beta:.3f} | Test AUC: {auc:.4f}")

    print("--- STAGE 1 Finished. VAE model is ready. ---")
    
    # Load the best model
    if os.path.exists(best_model_path):
        print(f"Loading best Stage 1 model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']+1} with test AUC: {checkpoint['test_auc']:.4f}")
    else:
        print("WARNING: No best model checkpoint found. Using final model state.")
    
    # Clean up optimizer memory
    del optimizer_vae
    torch.cuda.empty_cache()
    
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
            x = [torch.tensor(dataset_obj.x_s[i].to_dense())]
            x_tensor = torch.stack(x).to(device)
            features_dgl = x_tensor.view(-1, in_feature_dim)
            # NEW, CORRECTED line
            dgl_views = [dgl.batch([dgl_multiview_list[i][v]]).to(device) for v in range(args.num_views)]
            
            # Get graph-level embedding from VAE encoder
            _, graph_embedding, _ = vae_model.encoder(dgl_views, features_dgl, [1, x[0].shape[0]])

            # This is the key step: use the graph structure from one view
            # and prepare the node features for the DDM.
            dgl_graph_for_ddm = dgl_multiview_list[i][0] # Use first view's structure
            num_nodes = dgl_graph_for_ddm.num_nodes()

            # Broadcast the single graph-level embedding to all nodes
            node_features = graph_embedding.expand(num_nodes, -1)
            dgl_graph_for_ddm.ndata['attr'] = node_features.cpu()
            
            ddm_graphs.append((dgl_graph_for_ddm, dataset_obj.labels[i]))
            
        return ddm_graphs
        
    train_ddm_graphs = generate_ddm_input_dataset(vae_model, train_data, train_dgl_multiview)
    test_ddm_graphs = generate_ddm_input_dataset(vae_model, test_data, test_dgl_multiview)
    
    # ========================================================================
    # STAGE 2: TRAIN DDM FEATURE ENHANCER
    # ========================================================================
    print(f"\n--- STAGE 2: Training DDM Feature Enhancer ---")
    ddm_train_loader = DGLDataLoader(train_ddm_graphs, batch_size=args.batchSize, shuffle=True, collate_fn=collate_dgl, drop_last=True)
    ddm_test_loader = DGLDataLoader(test_ddm_graphs, batch_size=args.batchSize, shuffle=False, collate_fn=collate_dgl)

    ddm_main_args = {
        'in_dim': args.graphEmDim, # Input to DDM is the VAE embedding dimension
        'num_hidden': 256, 'num_layers': 3, 'nhead': 4, 'activation': 'gelu',
        'feat_drop': 0.2, 'attn_drop': 0.2, 'norm': 'layernorm'
    }
    ddm_kwargs = {
        'T': 500, 'beta_schedule': 'cosine', 'alpha_l': 2.0
    }
    ddm_model = DDM(**ddm_main_args, **ddm_kwargs).to(device)
    optimizer_ddm = torch.optim.Adam(ddm_model.parameters(), lr=5e-5, weight_decay=1e-5)

    best_val_auc = 0.0
    eval_T_values = [50, 100, 200, 300, 400]
    
    for epoch in range(800): # DDM may need fewer epochs
        ddm_model.train()
        total_loss = 0
        for batch_g, _ in tqdm(ddm_train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            batch_g = batch_g.to(device)
            features = batch_g.ndata['attr']
            optimizer_ddm.zero_grad()
            loss = ddm_model(batch_g, features)
            loss.backward()
            optimizer_ddm.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(ddm_train_loader)
        
        if 1==1:
            auc_mean, auc_std = evaluate_with_svm(ddm_model, ddm_test_loader, eval_T_values, device)
            print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | Test AUC (SVM): {auc_mean:.4f} ± {auc_std:.4f}")
            if auc_mean > best_val_auc:
                best_val_auc = auc_mean
                print(f"*** New best test AUC: {best_val_auc:.4f}. Saving model. ***")
                torch.save(ddm_model.state_dict(), f'ckpt/{args.dataset}/best_ddm_model.pth')

    print(f"\n--- Pipeline Finished. Best Test AUC with SVM: {best_val_auc:.4f} ---")