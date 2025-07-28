#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract detailed model information from checkpoint
"""

import os
import sys
import torch
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Extract model information from checkpoint')
    parser.add_argument('-model_path', type=str, default='/root/GraphVAE-MM/ckpt/Multi/best_ddm_model.pth', 
                        help='Path to the model checkpoint')
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Load the model checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Print basic information
        print("\n===== MODEL SUMMARY =====")
        print(f"Epoch: {checkpoint.get('epoch', 'Not available')}")
        print(f"Best AUC: {checkpoint.get('best_auc', 'Not available')}")
        print(f"Evaluation Timesteps: {checkpoint.get('eval_T_values', 'Not available')}")
        
        # Print arguments if available
        if 'args' in checkpoint:
            args_obj = checkpoint['args']
            print("\n===== TRAINING ARGUMENTS =====")
            for key in dir(args_obj):
                if not key.startswith('_'):
                    try:
                        value = getattr(args_obj, key)
                        if not callable(value):
                            print(f"{key}: {value}")
                    except:
                        pass
        
        # Print model architecture statistics
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"\n===== MODEL ARCHITECTURE =====")
            print(f"Total layers: {len(model_state)}")
            
            # Analyze layer types
            layer_types = defaultdict(int)
            param_shapes = {}
            total_params = 0
            
            for name, param in model_state.items():
                # Categorize layer types
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_type = parts[0] + '.' + parts[1]
                    layer_types[layer_type] += 1
                
                # Store parameter shapes
                param_shapes[name] = list(param.shape)
                
                # Count parameters
                total_params += param.numel()
            
            # Print layer type distribution
            print("\n----- Layer Type Distribution -----")
            for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
                print(f"{layer_type}: {count} layers")
            
            # Print parameter count
            print(f"\nTotal parameters: {total_params:,}")
            
            # Print key architectural components
            print("\n----- Key Architectural Components -----")
            
            # Check if it's a GAT-based architecture
            if any('gat' in k.lower() for k in model_state.keys()):
                print("Architecture: Graph Attention Network (GAT) based")
                
                # Check number of attention heads if applicable
                attn_heads = [v for k, v in param_shapes.items() if 'attention' in k.lower() and 'weight' in k]
                if attn_heads:
                    print(f"Attention heads: {attn_heads[0][0] if attn_heads else 'N/A'}")
            
            # Check if it's a GCN-based architecture
            elif any('gcn' in k.lower() for k in model_state.keys()):
                print("Architecture: Graph Convolutional Network (GCN) based")
            
            # Check if it's a Transformer-based architecture
            elif any('transformer' in k.lower() for k in model_state.keys()):
                print("Architecture: Transformer-based")
            
            # Check for diffusion model components
            if any('time_embedding' in k.lower() for k in model_state.keys()):
                print("Type: Denoising Diffusion Model (DDM)")
                time_embed_dims = [v for k, v in param_shapes.items() if 'time_embedding' in k and 'weight' in k]
                if time_embed_dims:
                    print(f"Time embedding dimension: {time_embed_dims[0][1] if time_embed_dims else 'N/A'}")
            
            # Identify embedding dimensions
            hidden_dims = set()
            for name, shape in param_shapes.items():
                if len(shape) == 2 and 'embedding' not in name.lower():
                    hidden_dims.add(shape[1])
            print(f"Hidden dimensions: {sorted(list(hidden_dims))}")
            
            # Print sample of important layers
            print("\n----- Sample Important Layers -----")
            important_layers = [k for k in model_state.keys() 
                               if any(x in k for x in ['embedding', 'attention', 'conv', 'gat', 'gcn', 'transformer'])]
            for i, layer in enumerate(important_layers[:10]):
                print(f"{layer}: {param_shapes[layer]}")
            
            if len(important_layers) > 10:
                print(f"... and {len(important_layers) - 10} more important layers")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
