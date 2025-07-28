#!/usr/bin/env python3
# simple_viz.py - Very simple t-SNE visualization for graph embeddings

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Import specific components
from data import list_graph_loader, Datasets
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

def visualize_tsne(embeddings, labels, perplexity=30, title="t-SNE Visualization"):
    """Apply t-SNE and visualize embeddings."""
    print(f"Running t-SNE with perplexity={perplexity}...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
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
    output_filename = f'ckpt/{args.dataset}/tsne_visualization.png'
    plt.savefig(output_filename, dpi=300)
    print(f"t-SNE visualization saved to {output_filename}")
    plt.show()

def visualize_pca(embeddings, labels, title="PCA Visualization"):
    """Apply PCA and visualize embeddings."""
    print(f"Running PCA...")
    
    # Apply PCA
    pca = PCA(n_components=2)
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
    output_filename = f'ckpt/{args.dataset}/pca_visualization.png'
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

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Simple t-SNE visualization')
    parser.add_argument('-dataset', dest="dataset", default="Multi")
    parser.add_argument('-device', dest="device", default="cuda:0")
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
                
        # Create simple t-SNE visualization of embeddings
        test_features = []
        test_labels = []
        
        print("\n--- Preparing data for visualization ---")
        # Collect features and labels
        for i in range(len(test_data)):
            # Process each graph
            features = test_data.x_s[i].to_dense()
            
            # If features are 3D (nodes, channels, features), flatten to 2D
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # Take the mean across nodes to get a graph-level representation
            graph_embedding = features.mean(dim=0)
            
            # Add to collection
            test_features.append(graph_embedding)
            test_labels.append(test_data.labels[i])
            
        # Convert to tensors
        test_features = torch.stack(test_features)
        test_labels = np.array(test_labels)
        
        print(f"Prepared data for visualization: {test_features.shape}")
        
        # Visualize with t-SNE
        print("\n--- Generating t-SNE Visualization ---")
        visualize_tsne(
            test_features.numpy(), 
            test_labels,
            perplexity=min(30, len(test_labels)-1),
            title="Graph Embedding t-SNE Visualization"
        )
        
        # Visualize with PCA
        print("\n--- Generating PCA Visualization ---")
        visualize_pca(
            test_features.numpy(),
            test_labels,
            title="Graph Embedding PCA Visualization"
        )
        
        # Evaluate embedding quality
        print("\n--- Evaluating Embedding Quality ---")
        evaluate_embedding_quality(test_features.numpy(), test_labels)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 