#!/usr/bin/env python3
# viz.py - Simple visualization script for trained GraphVAE-MM models

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
import scipy.sparse as sp

# Import specific components
from data import list_graph_loader, Datasets
from dgl.dataloading import GraphDataLoader as DGLDataLoader
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling
from sklearn.model_selection import train_test_split

def data_split(list_adj, list_x, list_label, test_size=0.2, random_state=None):
    """
    Splits the dataset into training and testing sets using sklearn.
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

def collate_dgl(batch):
    graphs, labels = map(list, zip(*batch))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def create_dgl_graphs(dataset_obj, num_views):
    dgl_graphs_list = []
    for i in range(len(dataset_obj.adj_s)):
        views = [dgl.from_scipy(sp.csr_matrix(dataset_obj.adj_s[i][v].cpu().numpy())) for v in range(num_views)]
        dgl_graphs_list.append(views)
    return dgl_graphs_list

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
        
        try:
            # Get denoised node features at this timestep
            denoised_nodes = model.embed(batch_g, feat, t_val)
            
            # Apply pooling strategies
            avg_embed = avg_pooler(batch_g, denoised_nodes)
            max_embed = max_pooler(batch_g, denoised_nodes)
            
            # Combine embeddings
            combined_embed = torch.cat([avg_embed, max_embed], dim=-1)
            
            # Store embeddings
            all_embeds.append(combined_embed.cpu().detach())
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    if not all_embeds:
        raise ValueError("Failed to extract any embeddings")
        
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
        elif method.lower() == 'pca':
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_embeds = pca.fit_transform(multi_ts_embeds[ts])
            method_name = 'PCA'
        else:
            raise ValueError(f"Unsupported method: {method}")
        
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
        
        plt.title(f"{method_name} Visualization at Timestep {ts}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.tight_layout()
    
    plt.suptitle(f"Embedding Visualization Comparison ({method.upper()})")
    plt.tight_layout()
    
    # Save figure
    output_filename = f'ckpt/{args.dataset}/embedding_comparison_{method}_{title.split()[1].lower()}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Embedding comparison visualization saved to {output_filename}")
    plt.show()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run visualization for trained models')
    parser.add_argument('-dataset', dest="dataset", default="Multi")
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-num_views', dest="num_views", default=2, type=int)
    parser.add_argument('-graphEmDim', dest="graphEmDim", default=512, type=int)
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
        list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)
        
        list_adj_train, list_adj_test, list_x_train, list_x_test, list_label_train, list_label_test = data_split(
            list_adj, list_x, list_label, test_size=0.2, random_state=42
        )
        
        train_data = Datasets(list_adj_train, True, list_x_train, list_label_train)
        test_data = Datasets(list_adj_test, True, list_x_test, list_label_test, Max_num=train_data.max_num_nodes)
        train_data.processALL(self_for_none=True)
        test_data.processALL(self_for_none=True)
        in_feature_dim = train_data.feature_size
        
        print(f"Dataset loaded. Train: {len(train_data)}, Test: {len(test_data)}")
        print(f"Input feature dimension: {in_feature_dim}")
        
        # Load the trained model
        best_model_path = f'ckpt/{args.dataset}/best_ddm_model.pth'
        print(f"Loading model from {best_model_path}")
        
        # Load model with weights_only=False to handle the error
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"Best AUC: {checkpoint.get('best_auc', 'N/A')}")
            
            # Extract saved state dict for visualization
            state_dict = checkpoint.get('ema_model_state_dict', checkpoint.get('model_state_dict', None))
            
            if state_dict is not None:
                # Save state dict as a new file that can be loaded with weights_only=True
                safe_path = f'ckpt/{args.dataset}/safe_model_weights.pth'
                torch.save(state_dict, safe_path)
                print(f"Saved safe model weights to {safe_path}")
                
                # Create embeddings directly from the state_dict
                print("\n--- Visualizing Embeddings ---")
                
                # Create simple t-SNE visualization of embeddings
                test_features = torch.cat([test_data.x_s[i].to_dense() for i in range(len(test_data))], dim=0)
                test_labels = np.array(test_data.labels)
                
                # Reshape features if needed
                if len(test_features.shape) > 2:
                    test_features = test_features.view(test_features.shape[0], -1)
                
                # Visualize raw features first
                visualize_tsne(
                    test_features.numpy(), 
                    test_labels,
                    perplexity=min(30, len(test_labels)-1),
                    title="Raw Feature Embeddings"
                )
                
                # Evaluate raw embedding quality
                print("\n--- Evaluating Raw Feature Embeddings ---")
                raw_sil, raw_db = evaluate_embedding_quality(test_features.numpy(), test_labels)
                
                # Visualize with K-means clustering
                visualize_kmeans_clusters(
                    test_features.numpy(), 
                    test_labels,
                    title="Raw Features Clustering"
                )
                
                # Create a PDF report
                print("\n--- Generating Report ---")
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
                        ['Raw Features', f"{raw_sil:.4f}", f"{raw_db:.4f}"]
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
                                    
                    plt.title('Embedding Quality Metrics', fontsize=16, pad=20)
                    pdf.savefig()
                    plt.close()
                
                print(f"Report saved to {output_filename}")
                
            else:
                print("No state dict found in the checkpoint")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 