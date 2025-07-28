#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of using visual.py to visualize embeddings from trained models.
This file demonstrates how to call the visualization functions from your main training script.
"""

import torch
import os
import argparse
from visual import visualize_models

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Example of using visualization with trained models')
    parser.add_argument('-vae_model_path', type=str, required=True, help='Path to the VAE model checkpoint')
    parser.add_argument('-ddm_model_path', type=str, required=True, help='Path to the DDM model checkpoint')
    parser.add_argument('-dataset', type=str, default="Multi", help='Dataset name')
    parser.add_argument('-device', type=str, default="cuda:0", help='Device to run on')
    parser.add_argument('-output_dir', type=str, default="visualizations", help='Output directory')
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example: After training your models and evaluating them, call the visualization
    # Code below is pseudo-code - you need to replace with your actual model loading code
    
    # --- Example: After training VAE model and DDM model ---
    # Load data and models (this is pseudo-code, replace with your actual code)
    # from model import StagedSupervisedVAE, ClassificationDecoder, DynamicCouplingEncoder
    # from data import list_graph_loader, Datasets
    
    # Load models
    print(f"Loading VAE model from {args.vae_model_path}...")
    vae_model = torch.load(args.vae_model_path, map_location=device)
    
    print(f"Loading DDM model from {args.ddm_model_path}...")
    ddm_model = torch.load(args.ddm_model_path, map_location=device)
    
    # --- Load dataset (pseudo-code) ---
    # train_data, test_data = load_your_dataset(args.dataset)
    # train_dgl_multiview, test_dgl_multiview = create_your_dgl_graphs(train_data, test_data)
    # ddm_train_loader, ddm_test_loader = create_your_data_loaders(train_data, test_data)
    
    # --- Set evaluation timesteps ---
    eval_timesteps = [100, 500, 900]  # Example timesteps for DDM evaluation
    
    # --- Call the visualization function ---
    visualize_models(
        vae_model=vae_model,
        ddm_model=ddm_model,
        train_data=train_data,
        test_data=test_data,
        train_dgl_multiview=train_dgl_multiview,
        test_dgl_multiview=test_dgl_multiview,
        ddm_train_loader=ddm_train_loader,
        ddm_test_loader=ddm_test_loader,
        eval_timesteps=eval_timesteps,
        device=device,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        use_test_data=True  # Use test data for visualization
    )
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 