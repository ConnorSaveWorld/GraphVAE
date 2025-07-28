#!/usr/bin/env python3
# run_viz.py - Visualization script for trained GraphVAE-MM models

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dgl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

# Import all functions from test.py
from test import (
    DDM, GraphTransformer, list_graph_loader, Datasets, data_split,
    DGLDataLoader, collate_dgl, AvgPooling, MaxPooling
)

def extract_embeddings(model, data_loader, t_val, device):
    """Extract graph-level embeddings from the DDM model."""
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
        with torch.no_grad():
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

@torch.no_grad()
def extract_transformer_embeddings(model, dataset_obj, dgl_multiview_list, device, in_feature_dim):
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
        _, graph_embedding, _ = model(combined_graph, features_dgl)
        
        # Store embeddings and labels
        all_embeds.append(graph_embedding.cpu())
        all_labels.append(dataset_obj.labels[i])
    
    # Convert to numpy arrays
    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    all_labels = np.array(all_labels)
    
    return all_embeds, all_labels

def visualize_tsne(embeddings, labels, perplexity=30, n_iter=1000, title="t-SNE Visualization of Graph Embeddings"):
    """Apply t-SNE and visualize embeddings."""
    print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
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
    output_filename = f'ckpt/{args.dataset}/tsne_visualization_{title.split()[1].lower()}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"t-SNE visualization saved to {output_filename}")
    plt.show()

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

def visualize_comparison(ddm_embeds, transformer_embeds, labels, perplexity=30, n_iter=1000, t_val=500):
    """Create a comparison plot of both embedding types."""
    plt.figure(figsize=(18, 8))
    
    # First, visualize transformer embeddings
    plt.subplot(1, 2, 1)
    tsne_transformer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
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
    tsne_ddm = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
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
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
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
    
    # Draw cluster centers
    centers_2d = tsne.transform(kmeans.cluster_centers_) if len(kmeans.cluster_centers_) > 0 else []
    ax2.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        marker='X', s=150, c='black', alpha=1.0,
        label='Centroids'
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

def save_visualization_report(ddm_sil, ddm_db, transformer_sil, transformer_db, args, t_val=500):
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

def create_dgl_graphs(dataset_obj, num_views):
    dgl_graphs_list = []
    for i in range(len(dataset_obj.adj_s)):
        views = [dgl.from_scipy(sp.csr_matrix(dataset_obj.adj_s[i][v].cpu().numpy())) for v in range(num_views)]
        dgl_graphs_list.append(views)
    return dgl_graphs_list

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run visualization for trained models')
    parser.add_argument('-dataset', dest="dataset", default="Multi")
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-num_views', dest="num_views", default=2, type=int)
    parser.add_argument('-graphEmDim', dest="graphEmDim", default=768, type=int)
    parser.add_argument('-t_val', dest="t_val", default=500, type=int, help="Timestep to use for DDM visualization")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(f'ckpt/{args.dataset}', exist_ok=True)

    try:
        # Load data
        print("Loading dataset...")
        import scipy.sparse as sp
        list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)
        
        list_adj_train, list_adj_test, list_x_train, list_x_test, list_label_train, list_label_test = data_split(
            list_adj, list_x, list_label, test_size=0.2, random_state=42
        )
        
        train_data = Datasets(list_adj_train, True, list_x_train, list_label_train)
        test_data = Datasets(list_adj_test, True, list_x_test, list_label_test, Max_num=train_data.max_num_nodes)
        train_data.processALL(self_for_none=True)
        test_data.processALL(self_for_none=True)
        in_feature_dim = train_data.feature_size
        
        # Create DGL graphs
        print("Creating DGL graphs...")
        train_dgl_multiview = create_dgl_graphs(train_data, args.num_views)
        test_dgl_multiview = create_dgl_graphs(test_data, args.num_views)
        
        # Generate DDM dataset from graph embeddings
        print("Processing test data for visualization...")
        test_ddm_graphs = []
        
        # Load models
        transformer_model_path = f'ckpt/{args.dataset}_stage1_transformer/best_stage1_transformer_model.pt'
        ddm_model_path = f'ckpt/{args.dataset}/best_ddm_model.pth'

        print(f"Loading Graph Transformer from {transformer_model_path}")
        transformer_model = GraphTransformer(
            in_feats=in_feature_dim,
            out_feats=args.graphEmDim,
            hidden_dim=1024,
            num_layers=3,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            num_classes=1
        ).to(device)
        
        try:
            transformer_checkpoint = torch.load(transformer_model_path, map_location=device, weights_only=False)
            transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
            print(f"Successfully loaded transformer model from epoch {transformer_checkpoint['epoch']+1}")
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            print("Continuing with untrained model")
        
        print(f"Loading DDM model from {ddm_model_path}")
        # Enhanced model configuration with increased capacity
        ddm_main_args = {
            'in_dim': args.graphEmDim,
            'num_hidden': 512,
            'num_layers': 2,
            'nhead': 8,
            'activation': 'gelu',
            'feat_drop': 0.1,
            'attn_drop': 0.1,
            'norm': 'layernorm'
        }
            
        # Better diffusion parameters
        ddm_kwargs = {
            'T': 1000,
            'beta_schedule': 'cosine',
            'alpha_l': 2.0,
            'beta_1': 5e-5,
            'beta_T': 0.02
        }
        
        ddm_model = DDM(**ddm_main_args, **ddm_kwargs).to(device)
        
        try:
            ddm_checkpoint = torch.load(ddm_model_path, map_location=device, weights_only=False)
            # Try loading EMA model first, fallback to regular model
            if ddm_checkpoint['ema_model_state_dict'] is not None:
                print("Using EMA model for visualization")
                ddm_model.load_state_dict(ddm_checkpoint['ema_model_state_dict'])
            else:
                print("Using regular model for visualization")
                ddm_model.load_state_dict(ddm_checkpoint['model_state_dict'])
            print(f"Successfully loaded DDM model from epoch {ddm_checkpoint['epoch']+1}")
        except Exception as e:
            print(f"Error loading DDM model: {e}")
            print("Continuing with untrained model")
            
        # Generate DDM-ready dataset
        print("Generating DDM-ready dataset...")
        
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

                # Store processed features
                dgl_graph_for_ddm.ndata['attr'] = node_embeddings.cpu()
                ddm_graphs.append((dgl_graph_for_ddm, dataset_obj.labels[i]))
                
            return ddm_graphs
        
        # Generate datasets for DDM
        test_ddm_graphs = generate_ddm_input_dataset(transformer_model, test_data, test_dgl_multiview)
        
        # Create data loader with minimal overhead
        ddm_test_loader = DGLDataLoader(
            test_ddm_graphs, 
            batch_size=8,  # Use small batch size to avoid OOM
            shuffle=False, 
            collate_fn=collate_dgl,
            num_workers=0  # No workers to avoid memory duplication
        )
        
        # Set the timestep for visualization
        t_val = args.t_val
        print(f"\n--- Starting Visualization (using timestep t={t_val}) ---")
        
        # Extract DDM embeddings
        print("Extracting DDM embeddings...")
        embeddings, labels = extract_embeddings(ddm_model, ddm_test_loader, t_val, device)
        
        # Visualize using t-SNE
        print(f"Shape of DDM embeddings: {embeddings.shape}, labels: {labels.shape}")
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
            ts_embeds, _ = extract_embeddings(ddm_model, ddm_test_loader, ts, device)
            multi_timestep_embeds[ts] = ts_embeds
        
        # Visualize embeddings at different timesteps
        print("\n--- Visualizing Embeddings at Different Timesteps ---")
        visualize_timestep_comparison(multi_timestep_embeds, labels, method='tsne')
        visualize_timestep_comparison(multi_timestep_embeds, labels, method='pca')
        
        # Also try PCA for the main embeddings
        print("\n--- Visualizing with PCA instead of t-SNE ---")
        visualize_pca(embeddings, labels, title=f"PCA of DDM Embeddings")
        
        # Extract transformer embeddings
        print("\n--- Extracting Graph Transformer Embeddings ---")
        transformer_embeds, transformer_labels = extract_transformer_embeddings(
            transformer_model, test_data, test_dgl_multiview, device, in_feature_dim
        )
        
        # Visualize transformer embeddings
        print(f"Shape of transformer embeddings: {transformer_embeds.shape}")
        visualize_tsne(transformer_embeds, transformer_labels, 
                      perplexity=min(30, len(transformer_labels)-1),
                      title="t-SNE Visualization of Graph Transformer Embeddings")
        visualize_pca(transformer_embeds, transformer_labels, title="PCA of Transformer")
        
        # Create comparison visualization
        visualize_comparison(embeddings, transformer_embeds, labels, perplexity=min(30, len(labels)-1), t_val=t_val)
        
        # Evaluate embedding quality
        print("\n--- Evaluating Graph Transformer Embeddings ---")
        transformer_sil, transformer_db = evaluate_embedding_quality(transformer_embeds, transformer_labels)
        
        print("\n--- Evaluating DDM Enhanced Embeddings ---")
        ddm_sil, ddm_db = evaluate_embedding_quality(embeddings, labels)
        
        # Visualize clustering for both embedding types
        print("\n--- Visualizing K-means Clustering on Embeddings ---")
        ddm_clusters = visualize_kmeans_clusters(embeddings, labels, 
                                               title=f"DDM Embeddings (t={t_val})")
        transformer_clusters = visualize_kmeans_clusters(transformer_embeds, transformer_labels,
                                                        title="Transformer Embeddings")
        
        # Generate the report
        save_visualization_report(ddm_sil, ddm_db, transformer_sil, transformer_db, args, t_val)
        
        print("\n--- Visualization Complete! ---")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc() 