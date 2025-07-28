#!/usr/bin/env python3
"""Test script for enhanced t-SNE visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import math

# Mock args for testing
class Args:
    dataset = "Multi"

args = Args()

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
    """Enhanced t-SNE visualisation with multiple improvements for small datasets."""
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
            
            tsne = TSNE(
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
        tsne = TSNE(
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
    
    plt.title(f"{title}\nSeparation Score: {best_separation:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    if hide_axis:
        plt.axis('off')
    
    plt.tight_layout()
    print(f"Enhanced t-SNE visualization would be saved to ckpt/{args.dataset}/{save_name}")
    plt.savefig(f'ckpt/{args.dataset}/{save_name}', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_embeddings_2d, best_separation

# Generate test data similar to your case
np.random.seed(42)

# Create mock embeddings similar to your data dimensions
n_samples = 88
n_features = 384  # Post-diffusion embedding size
n_pos = 21
n_neg = 67

# Generate embeddings with some structure
embeddings = np.random.randn(n_samples, n_features)

# Add some class-specific structure
labels = np.array([1] * n_pos + [0] * n_neg)

# Add class-specific bias to make classes somewhat separable
embeddings[:n_pos] += np.random.randn(1, n_features) * 0.5  # Positive class bias
embeddings[n_pos:] += np.random.randn(1, n_features) * 0.3  # Negative class bias

print("Testing enhanced t-SNE visualization...")
print(f"Data shape: {embeddings.shape}")
print(f"Class distribution: {n_pos} positive, {n_neg} negative")

# Test the enhanced function
try:
    import os
    os.makedirs('ckpt/Multi', exist_ok=True)
    
    result_embeddings, separation = visualize_tsne(
        embeddings, labels,
        title="Enhanced t-SNE Test",
        save_name="test_enhanced_tsne.png"
    )
    
    print(f"\n✓ Enhanced t-SNE completed successfully!")
    print(f"✓ Final separation score: {separation:.4f}")
    
    if separation > 0.1:
        print("✓ Good separation achieved!")
    elif separation > 0.0:
        print("⚠ Moderate separation")
    else:
        print("⚠ Poor separation")
        
except Exception as e:
    print(f"✗ Enhanced t-SNE failed: {e}")
    import traceback
    traceback.print_exc()
