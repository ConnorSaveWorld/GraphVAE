#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified visualization script for DDM model checkpoint
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Visualize DDM model weights using t-SNE')
    parser.add_argument('-model_path', type=str, default='/root/GraphVAE-MM/ckpt/Multi/best_ddm_model.pth', 
                        help='Path to the DDM model checkpoint')
    parser.add_argument('-output_dir', type=str, default='visualizations', 
                        help='Directory to save visualizations')
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Load the model checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint loaded. Available keys: {list(checkpoint.keys())}")
        
        # Extract the model state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"Model state dict contains {len(model_state)} layers")
            
            # Extract weights from the model
            weight_tensors = []
            names = []
            for name, param in model_state.items():
                if 'weight' in name and param.dim() == 2 and param.size(0) > 10 and param.size(1) > 10:
                    print(f"Adding layer {name} with shape {param.shape}")
                    weight_tensors.append(param.detach().cpu().numpy())
                    names.append(name)
            
            if not weight_tensors:
                print("No suitable weight tensors found for visualization")
                return
                
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Visualize individual weight matrices
            for i, (weight, name) in enumerate(zip(weight_tensors, names)):
                print(f"Visualizing {name}...")
                
                # Flatten the weight matrix for t-SNE
                flattened = weight.reshape(weight.shape[0], -1)
                
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, flattened.shape[0]-1))
                embedded = tsne.fit_transform(flattened)
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                plt.scatter(embedded[:, 0], embedded[:, 1], c=np.arange(embedded.shape[0]), cmap='viridis', alpha=0.8)
                plt.colorbar(label='Row Index')
                plt.title(f"t-SNE of {name} weights")
                plt.tight_layout()
                
                # Save the figure
                output_path = os.path.join(args.output_dir, f"{name.replace('.', '_')}_tsne.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization to {output_path}")
            
            # Also visualize the EMA weights if available
            if 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
                print("Processing EMA weights...")
                ema_state = checkpoint['ema_model_state_dict']
                
                # Extract weights from the EMA model
                ema_weight_tensors = []
                ema_names = []
                for name, param in ema_state.items():
                    if 'weight' in name and param.dim() == 2 and param.size(0) > 10 and param.size(1) > 10:
                        print(f"Adding EMA layer {name} with shape {param.shape}")
                        ema_weight_tensors.append(param.detach().cpu().numpy())
                        ema_names.append(f"ema_{name}")
                
                # Visualize individual EMA weight matrices
                for i, (weight, name) in enumerate(zip(ema_weight_tensors, ema_names)):
                    print(f"Visualizing {name}...")
                    
                    # Flatten the weight matrix for t-SNE
                    flattened = weight.reshape(weight.shape[0], -1)
                    
                    # Apply t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, flattened.shape[0]-1))
                    embedded = tsne.fit_transform(flattened)
                    
                    # Create visualization
                    plt.figure(figsize=(10, 8))
                    plt.scatter(embedded[:, 0], embedded[:, 1], c=np.arange(embedded.shape[0]), cmap='viridis', alpha=0.8)
                    plt.colorbar(label='Row Index')
                    plt.title(f"t-SNE of {name} weights")
                    plt.tight_layout()
                    
                    # Save the figure
                    output_path = os.path.join(args.output_dir, f"{name.replace('.', '_')}_tsne.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved visualization to {output_path}")
        else:
            print("No model state dictionary found in checkpoint")
            
    except Exception as e:
        print(f"Error loading or processing model: {e}")
        import traceback
        traceback.print_exc()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 