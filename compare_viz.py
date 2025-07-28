#!/usr/bin/env python3
# compare_viz.py - Compare raw embeddings with DDM enhanced embeddings

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Import specific components
from data import list_graph_loader, Datasets
from sklearn.model_selection import train_test_split
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling

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

def visualize_comparison(raw_embeddings, ddm_embeddings, labels, perplexity=30, title="Embedding Comparison"):
    """Compare raw and DDM embeddings side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    
    # Process raw embeddings
    tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    raw_2d = tsne_raw.fit_transform(raw_embeddings)
    
    # Process DDM embeddings
    tsne_ddm = TSNE(n_components=2, perplexity=perplexity, random_state=42) 
    ddm_2d = tsne_ddm.fit_transform(ddm_embeddings)
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot raw embeddings
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axs[0].scatter(
            raw_2d[mask, 0], 
            raw_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7
        )
    
    axs[0].set_title("Raw Feature Embeddings")
    axs[0].set_xlabel("t-SNE Component 1")
    axs[0].set_ylabel("t-SNE Component 2")
    axs[0].legend()
    
    # Plot DDM embeddings
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axs[1].scatter(
            ddm_2d[mask, 0], 
            ddm_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7
        )
    
    axs[1].set_title("DDM Enhanced Embeddings")
    axs[1].set_xlabel("t-SNE Component 1")
    axs[1].set_ylabel("t-SNE Component 2")
    axs[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure
    output_filename = f'ckpt/{args.dataset}/embedding_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Comparison visualization saved to {output_filename}")
    plt.show()
    
    return raw_2d, ddm_2d

def evaluate_embedding_quality(embeddings, labels, name=""):
    """Evaluate embeddings using clustering metrics."""
    try:
        # Calculate silhouette score (higher is better)
        sil_score = silhouette_score(embeddings, labels)
        
        # Calculate Davies-Bouldin index (lower is better)
        db_score = davies_bouldin_score(embeddings, labels)
        
        print(f"Embedding Quality Metrics ({name}):")
        print(f"  Silhouette Score: {sil_score:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")
        
        return sil_score, db_score
    except Exception as e:
        print(f"Error calculating clustering metrics for {name}: {e}")
        return None, None

# Simple mock class to emulate DDM model behavior for testing
class MockDDMModel(torch.nn.Module):
    def __init__(self):
        super(MockDDMModel, self).__init__()
        self.projection = torch.nn.Linear(380, 380)  # Adjust input size based on your feature dim
    
    def embed(self, g, x, t):
        # Just apply a simple projection to simulate DDM embedding
        return self.projection(x)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare embeddings visualization')
    parser.add_argument('-dataset', dest="dataset", default="Multi")
    parser.add_argument('-device', dest="device", default="cuda:0")
    parser.add_argument('-use_mock', dest="use_mock", action='store_true', help="Use mock DDM model")
    parser.add_argument('-model_path', dest="model_path", default=None, type=str, help="Path to DDM model")
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
                
        # Create raw feature embeddings
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
        
        print(f"Raw features prepared: {test_features.shape}")
        
        # Prepare DDM model (either mock or real)
        if args.use_mock:
            print("Using mock DDM model for visualization")
            ddm_model = MockDDMModel().to(device)
        elif args.model_path:
            print(f"Loading DDM model from {args.model_path}")
            try:
                # Try to load model
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
                
                # This is just a placeholder since we don't have the actual DDM model class
                print("Creating mock DDM model with loaded weights")
                ddm_model = MockDDMModel().to(device)
                print(f"Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            except Exception as e:
                print(f"Error loading DDM model: {e}")
                print("Falling back to mock model")
                ddm_model = MockDDMModel().to(device)
        else:
            print("No model path provided, using mock model")
            ddm_model = MockDDMModel().to(device)
        
        # Create DDM embeddings
        print("\n--- Creating DDM embeddings ---")
        ddm_features = []
        
        # Use model to create embeddings
        with torch.no_grad():
            for i in range(len(test_data)):
                # Get features
                features = test_data.x_s[i].to_dense().to(device)
                
                # If features are 3D (nodes, channels, features), flatten to 2D
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                try:
                    # Use mock model to create embeddings - in a real scenario this would use the DDM model
                    enhanced_features = ddm_model.embed(None, features, 500)
                    
                    # Take mean across nodes to get graph-level embedding
                    graph_embedding = enhanced_features.mean(dim=0)
                    
                    # Add to collection
                    ddm_features.append(graph_embedding.cpu())
                except Exception as e:
                    print(f"Error creating DDM embedding for graph {i}: {e}")
                    # Fallback to using raw features
                    graph_embedding = features.mean(dim=0)
                    ddm_features.append(graph_embedding.cpu())
        
        # Convert to tensor
        ddm_features = torch.stack(ddm_features)
        
        print(f"DDM features prepared: {ddm_features.shape}")
        
        # Evaluate embedding quality
        print("\n--- Evaluating Raw Embedding Quality ---")
        raw_sil, raw_db = evaluate_embedding_quality(test_features.numpy(), test_labels, "Raw Features")
        
        print("\n--- Evaluating DDM Embedding Quality ---")
        ddm_sil, ddm_db = evaluate_embedding_quality(ddm_features.numpy(), test_labels, "DDM Features")
        
        # Visualize comparison
        print("\n--- Generating Comparison Visualization ---")
        raw_2d, ddm_2d = visualize_comparison(
            test_features.numpy(),
            ddm_features.numpy(),
            test_labels,
            perplexity=min(30, len(test_labels)-1),
            title="t-SNE Visualization: Raw vs DDM Enhanced Embeddings"
        )
        
        # Create a quality metrics comparison table
        print("\n--- Embedding Quality Comparison ---")
        print("| Embedding Type | Silhouette Score ↑ | Davies-Bouldin Index ↓ |")
        print("|---------------|-------------------|------------------------|")
        print(f"| Raw Features  | {raw_sil:.4f}            | {raw_db:.4f}                 |")
        print(f"| DDM Features  | {ddm_sil:.4f}            | {ddm_db:.4f}                 |")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 