import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import dgl
import scipy.sparse as sp
from tqdm import tqdm

# Import necessary functions and classes from the main script
from newMain4 import DDM, list_graph_loader, Datasets, data_split
from data import list_graph_loader, Datasets
from dgl.dataloading import GraphDataLoader as DGLDataLoader
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set

def collate_dgl(batch):
    """Collate function for DGL dataloader."""
    graphs, labels = map(list, zip(*batch))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def create_dgl_graphs(dataset_obj, num_views=2):
    """Create DGL graphs from the dataset."""
    dgl_graphs_list = []
    for i in range(len(dataset_obj.adj_s)):
        views = [dgl.from_scipy(sp.csr_matrix(dataset_obj.adj_s[i][v].cpu().numpy())) for v in range(num_views)]
        dgl_graphs_list.append(views)
    return dgl_graphs_list

def extract_embeddings(model, dataloader, device, timesteps=[500]):
    """Extract embeddings from the model at specific timesteps."""
    model.eval()
    
    # Initialize pooling operations
    avg_pooler = AvgPooling()
    max_pooler = MaxPooling()
    
    # Simple attention pooling
    if hasattr(model.net, 'num_hidden'):
        feature_dim = model.net.num_hidden
    else:
        feature_dim = 768
    
    attn_gate_nn = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1)
    ).to(device)
    attn_pooler = GlobalAttentionPooling(attn_gate_nn)
    
    # Storage for embeddings and labels
    embeddings_by_t = {t: [] for t in timesteps}
    all_labels = []
    
    # Process each batch
    with torch.no_grad():
        for batch_g, labels in tqdm(dataloader, desc="Extracting embeddings"):
            batch_g, labels = batch_g.to(device), labels.to(device)
            feat = batch_g.ndata['attr']
            all_labels.append(labels.cpu())
            
            # Process each timestep
            for t_val in timesteps:
                # Get denoised node features at this timestep
                denoised_nodes = model.embed(batch_g, feat, t_val)
                
                # Apply pooling strategies to get graph-level embeddings
                avg_embed = avg_pooler(batch_g, denoised_nodes)
                max_embed = max_pooler(batch_g, denoised_nodes)
                attn_embed = attn_pooler(batch_g, denoised_nodes)
                
                # Combine the embeddings
                combined_embed = torch.cat([avg_embed, max_embed, attn_embed], dim=-1)
                embeddings_by_t[t_val].append(combined_embed.cpu())
    
    # Concatenate all embeddings and labels
    all_labels = torch.cat(all_labels, dim=0).numpy()
    final_embeds = {t: torch.cat(embeds, dim=0).numpy() for t, embeds in embeddings_by_t.items()}
    
    return final_embeds, all_labels

def main():
    parser = argparse.ArgumentParser(description='Visualize brain graph embeddings using t-SNE')
    parser.add_argument('--model_path', default='/root/GraphVAE-MM/ckpt/Multi/best_ddm_model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--dataset_path', default='/root/GraphVAE-MM/dataset/Multi', 
                        help='Path to the dataset')
    parser.add_argument('--dataset', default='Multi', help='Dataset name')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    parser.add_argument('--num_views', default=2, type=int, help='Number of views')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)
    
    # Split the dataset
    list_adj_train, list_adj_test, list_x_train, list_x_test, list_label_train, list_label_test = data_split(
        list_adj, list_x, list_label, test_size=0.2, random_state=42
    )
    
    # Create dataset objects
    train_data = Datasets(list_adj_train, True, list_x_train, list_label_train)
    test_data = Datasets(list_adj_test, True, list_x_test, list_label_test, Max_num=train_data.max_num_nodes)
    train_data.processALL(self_for_none=True)
    test_data.processALL(self_for_none=True)
    in_feature_dim = train_data.feature_size
    
    # Create DGL graphs
    print("Creating DGL graphs...")
    train_dgl_multiview = create_dgl_graphs(train_data, args.num_views)
    test_dgl_multiview = create_dgl_graphs(test_data, args.num_views)
    
    # Prepare DDM-ready datasets
    print("Preparing datasets for DDM...")
    train_ddm_graphs = []
    for i in range(len(train_data)):
        dgl_graph = train_dgl_multiview[i][0]  # Use first view's structure
        # For visualization purposes, we'll use the node features directly
        node_features = train_data.x_s[i].to_dense()
        node_features = torch.nn.functional.normalize(node_features, dim=1)  # Normalize features
        dgl_graph.ndata['attr'] = node_features
        train_ddm_graphs.append((dgl_graph, train_data.labels[i]))
    
    test_ddm_graphs = []
    for i in range(len(test_data)):
        dgl_graph = test_dgl_multiview[i][0]
        node_features = test_data.x_s[i].to_dense()
        node_features = torch.nn.functional.normalize(node_features, dim=1)
        dgl_graph.ndata['attr'] = node_features
        test_ddm_graphs.append((dgl_graph, test_data.labels[i]))
    
    # Create dataloaders
    train_loader = DGLDataLoader(
        train_ddm_graphs, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_dgl, drop_last=False
    )
    
    test_loader = DGLDataLoader(
        test_ddm_graphs, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_dgl, drop_last=False
    )
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Initialize the model with the same parameters used during training
    model_args = {
        'in_dim': 512,  # Changed to match the checkpoint (assuming 512 was used in training)
        'num_hidden': 512,
        'out_dim': 512,  # Add this to explicitly set the output dimension
        'num_layers': 2,
        'nhead': 8,
        'activation': 'gelu',
        'feat_drop': 0.1,
        'attn_drop': 0.1,
        'norm': 'layernorm',
        'T': 1000,
        'beta_schedule': 'cosine',
        'alpha_l': 2.0,
        'beta_1': 5e-5,
        'beta_T': 0.02
    }
    
    # Create model and load state dict
    ddm_model = DDM(**model_args).to(device)
    
    # Print model architecture for debugging
    print(f"Model initialized with hidden_dim={model_args['num_hidden']} and layers={model_args['num_layers']}")
    
    # Try loading either the EMA model (preferred) or the regular model
    if checkpoint['ema_model_state_dict'] is not None:
        print("Loading EMA model state...")
        ddm_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("Loading regular model state...")
        ddm_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with best AUC: {checkpoint['best_auc']:.4f}")
    
    # Extract embeddings at different timesteps for visualization
    timesteps = [100, 500, 900]
    print(f"Extracting embeddings at timesteps: {timesteps}")
    
    # Extract embeddings from train and test datasets
    train_embeddings, train_labels = extract_embeddings(ddm_model, train_loader, device, timesteps)
    test_embeddings, test_labels = extract_embeddings(ddm_model, test_loader, device, timesteps)
    
    # Create t-SNE visualizations for each timestep
    for t in timesteps:
        print(f"Creating t-SNE visualization for timestep {t}...")
        
        # Apply t-SNE to train embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(train_labels)-1))
        train_tsne = tsne.fit_transform(train_embeddings[t])
        
        # Apply t-SNE to test embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_labels)-1))
        test_tsne = tsne.fit_transform(test_embeddings[t])
        
        # Plot train embeddings
        plt.figure(figsize=(12, 10))
        
        # Create color map based on unique labels
        unique_labels = np.unique(train_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class with a different color
        for i, label in enumerate(unique_labels):
            mask = train_labels == label
            plt.scatter(
                train_tsne[mask, 0], train_tsne[mask, 1],
                c=[colors[i]], label=f'Class {label}',
                alpha=0.7, edgecolors='w', linewidth=0.5
            )
        
        plt.title(f't-SNE Visualization of Train Embeddings (Timestep {t})')
        plt.legend()
        plt.savefig(f'visualizations/train_tsne_t{t}.png', dpi=300, bbox_inches='tight')
        
        # Plot test embeddings
        plt.figure(figsize=(12, 10))
        
        # Create color map based on unique labels
        unique_labels = np.unique(test_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class with a different color
        for i, label in enumerate(unique_labels):
            mask = test_labels == label
            plt.scatter(
                test_tsne[mask, 0], test_tsne[mask, 1],
                c=[colors[i]], label=f'Class {label}',
                alpha=0.7, edgecolors='w', linewidth=0.5
            )
        
        plt.title(f't-SNE Visualization of Test Embeddings (Timestep {t})')
        plt.legend()
        plt.savefig(f'visualizations/test_tsne_t{t}.png', dpi=300, bbox_inches='tight')
        
        # Combined visualization (train + test)
        plt.figure(figsize=(12, 10))
        
        # Combine embeddings and labels
        all_tsne = np.vstack([train_tsne, test_tsne])
        all_labels = np.concatenate([train_labels, test_labels])
        dataset_indicator = np.concatenate([np.zeros_like(train_labels), np.ones_like(test_labels)])
        
        # Create markers for train/test
        markers = ['o', 's']  # circle for train, square for test
        
        # Plot each class with a different color, and train/test with different markers
        for i, label in enumerate(unique_labels):
            for j, (name, indicator) in enumerate([('Train', 0), ('Test', 1)]):
                mask = (all_labels == label) & (dataset_indicator == indicator)
                plt.scatter(
                    all_tsne[mask, 0], all_tsne[mask, 1],
                    c=[colors[i]], marker=markers[j],
                    label=f'{name} - Class {label}' if i == 0 else f'Class {label}',
                    alpha=0.7, edgecolors='w', linewidth=0.5
                )
        
        plt.title(f't-SNE Visualization of All Embeddings (Timestep {t})')
        plt.legend()
        plt.savefig(f'visualizations/combined_tsne_t{t}.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
    
    print("Visualization completed. Results saved in 'visualizations/' directory.")

if __name__ == '__main__':
    main()
