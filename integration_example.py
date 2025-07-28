# Add this at the end of newMain4.py, right after the pipeline finished message

# --- Visualization section ---
if args.visualize:
    try:
        # Import the visualization module
        from visual import visualize_models
        
        print("\n--- Creating t-SNE Visualizations ---")
        
        # Call the visualization function with your trained models and data
        visualize_models(
            vae_model=vae_model,  # Your trained VAE model
            ddm_model=ddm_model,  # Your trained DDM model
            train_data=train_data,
            test_data=test_data,
            train_dgl_multiview=train_dgl_multiview,
            test_dgl_multiview=test_dgl_multiview,
            ddm_train_loader=ddm_train_loader,
            ddm_test_loader=ddm_test_loader,
            eval_timesteps=eval_T_values,  # Use the same timesteps as in evaluation
            device=device,
            dataset_name=args.dataset,
            output_dir="visualizations",
            use_test_data=True  # Use test data for visualization
        )
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing without visualizations...")

# --- End of script --- 