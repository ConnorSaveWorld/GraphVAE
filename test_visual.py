#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to diagnose visual.py issues
"""

import os
import sys
import torch
import pickle

def main():
    print("Starting test script...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    model_path = "/root/GraphVAE-MM/ckpt/Multi/best_ddm_model.pth"
    print(f"Checking if model exists: {os.path.exists(model_path)}")
    
    print("Attempting to load model with pickle...")
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Successfully loaded with pickle. Keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading with pickle: {e}")
    
    print("Attempting to load model with torch.load and weights_only=False...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded with torch.load and weights_only=False. Keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading with torch.load and weights_only=False: {e}")
    
    print("Test complete!")

if __name__ == "__main__":
    main() 