#
# newMain.py (Modified for Two-Stage Supervised Pre-training)
#
import logging
import plotter
import torch.nn.functional as F
import argparse
from model import * # <<< Ensure this imports StagedSupervisedVAE, ClassificationDecoder
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl
import scipy.sparse as sp
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
from torchmetrics.classification import Accuracy, F1Score, AUROC

# <<< MODIFIED: Removed unused imports to clean up
from label_diffusion import LabelDiffusionClassifier
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
import torch._dynamo

# <<< MODIFIED: Set a higher default epoch number for meaningful training
parser = argparse.ArgumentParser(description='Supervised VAE Pre-training')
parser.add_argument('-e', dest="epoch_number", default=100, help="Number of Epochs for Stage 1 pre-training", type=int)
parser.add_argument('-lr', dest="lr", default=1e-4, help="Model learning rate", type=float) # <<< MODIFIED: Lower default LR
parser.add_argument('-batchSize', dest="batchSize", default=8, help="The size of each batch", type=int) # <<< MODIFIED: Smaller default BS
parser.add_argument('-device', dest="device", default="cuda:3", help="Which device should be used") # <<< MODIFIED: Default to cuda:0
parser.add_argument('-graphEmDim', dest="graphEmDim", default=128, help="The dimension of graph latent space", type=int) # <<< MODIFIED: Smaller latent dim
# --- Other arguments can remain as they are ---
parser.add_argument('-v', dest="Vis_step", default=1000, help="at every Vis_step 'minibatch' the plots will be updated")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-dataset', dest="dataset", default="Multi", help="Dataset name")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None, help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-PATH', dest="PATH", default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-encoder_type', dest="encoder_type", default="DynamicCoupling", help="Encoder type")
parser.add_argument('-UseGPU', dest="UseGPU", default=True, help="either use GPU or not if availabel")
parser.add_argument('-num_views', dest="num_views", default=2, help="Number of views in the multi-view graph dataset", type=int)
parser.add_argument('-task', dest="task", default="graphClassification", help="options: graphGeneration, graphClassification")
# ... (any other args you need) ...
args = parser.parse_args()

# --- SEEDING and SETUP (Unchanged) ---
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(args.device if torch.cuda.is_available() and args.UseGPU else "cpu")
print(f"--- Using device: {device} ---")
# ... (logging setup is fine) ...

def SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta, kl_threshold=0.5):
    classif_loss = F.binary_cross_entropy_with_logits(
        predicted_logits, target_labels.float().view(-1, 1)
    )
    kl_div = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - log_std.exp().pow(2), dim=1)
    kl_loss = torch.mean(torch.clamp(kl_div, min=kl_threshold)) # Free bits
    total_loss = classif_loss + (kl_beta * kl_loss)
    return total_loss, classif_loss, kl_loss

# --- DATA LOADING (Unchanged) ---
print("--- Loading Raw Data ---")
list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)
list_adj, test_list_adj, list_x_train, list_x_test, list_label_train, list_label_test = data_split(list_adj, list_x, list_label)

list_graphs = Datasets(list_adj, True, list_x_train, list_label_train)
list_test_graphs = Datasets(test_list_adj, True, list_x_test, list_label_test, Max_num=list_graphs.max_num_nodes)

print("Processing training and testing data...")
list_graphs.processALL(self_for_none=True)
list_test_graphs.processALL(self_for_none=True)
in_feature_dim = list_graphs.feature_size
print(f"--- Data Processed. Input feature dimension: {in_feature_dim} ---")

# ========================================================================
#
#                           STAGE 1: SUPERVISED PRE-TRAINING
#
# ========================================================================
print("\n--- STAGE 1: Setting up Supervised VAE for Pre-training ---")

# <<< MODIFIED: Use GraphConv in the encoder for efficiency >>>
encoder = DynamicCouplingEncoder(
    in_feature_dim=in_feature_dim,
    num_views=args.num_views,
    hidden_dim=256,
    num_initial_layers=2,
    num_coupling_layers=2,
    dim_coupling=128,
    dropout_rate=0.2, # Slightly increased dropout for regularization
    GraphLatentDim=args.graphEmDim
)

num_classes = 1 # For binary classification
decoder = ClassificationDecoder(
    latent_dim=args.graphEmDim,
    hidden_dim=512,
    num_classes=num_classes
)

model = StagedSupervisedVAE(encoder, decoder).to(device)
print(model)
logging.info(str(model))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
kl_beta = 0.0
kl_anneal_epochs = 80 # Anneal over more epochs
steps_per_epoch = max(1, len(list_graphs.list_adjs) // args.batchSize)
kl_anneal_steps = steps_per_epoch * kl_anneal_epochs
kl_anneal_rate = 1.0 / kl_anneal_steps if kl_anneal_steps > 0 else 1.0
print(f"KL Annealing will occur over the first {kl_anneal_epochs} epochs.")

print("\n--- Starting Stage 1: Supervised Pre-training ---")
for epoch in range(args.epoch_number):
    model.train()
    list_graphs.shuffle()
    
    epoch_total_loss, epoch_class_loss, epoch_kl_loss = 0, 0, 0
    num_batches = 0

    for i in range(0, len(list_graphs.list_adjs), args.batchSize):
        from_ = i
        to_ = i + args.batchSize
        
        # <<< MODIFIED: Ensure your get__ method can return labels >>>
        adj_batch, x_batch, _, _, _, labels_batch = list_graphs.get__(from_, to_, self_for_none=True, get_labels=True)
        target_labels = torch.tensor(labels_batch, device=device)

        # Prepare inputs for the model
        x_s_tensor = torch.stack(x_batch).to(device)
        features_for_dgl = x_s_tensor.view(-1, in_feature_dim)
        
        dgl_graphs_per_view = []
        for v in range(args.num_views):
            view_graphs_in_batch = [dgl.from_scipy(sp.csr_matrix(g[v].cpu().numpy())) for g in adj_batch]
            dgl_graphs_per_view.append(dgl.batch(view_graphs_in_batch).to(device))
        
        batchSize_info = [len(adj_batch), adj_batch[0].shape[-1]]

        # Update KL Beta
        if kl_beta < 1.0:
            kl_beta = min(1.0, kl_beta + kl_anneal_rate)

        # Forward pass, loss calculation, backward pass
        optimizer.zero_grad()
        predicted_logits, mean, log_std, _ = model(dgl_graphs_per_view, features_for_dgl, batchSize_info)
        total_loss, class_loss, kl_loss = SupervisedVAELoss(predicted_logits, target_labels, mean, log_std, kl_beta)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_class_loss += class_loss.item()
        epoch_kl_loss += kl_loss.item()
        num_batches += 1

    # --- End of Epoch Logging and Evaluation for Stage 1 ---
    avg_total_loss = epoch_total_loss / num_batches
    avg_class_loss = epoch_class_loss / num_batches
    avg_kl_loss = epoch_kl_loss / num_batches
    
    # Evaluate on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
    # Iterate over the test set in mini-batches
        for i_test in range(0, len(list_test_graphs.list_adjs), args.batchSize):
            from_test = i_test
            to_test = i_test + args.batchSize
        
        # Get a mini-batch of test data
            adj_test, x_test, _, _, _, labels_test = list_test_graphs.get__(from_test, to_test, self_for_none=True, get_labels=True)
        
        # Prepare DGL graphs for this mini-batch
            x_s_tensor_test = torch.stack(x_test).to(device)
            features_dgl_test = x_s_tensor_test.view(-1, in_feature_dim)
            dgl_views_test = [dgl.batch([dgl.from_scipy(sp.csr_matrix(g[v].cpu().numpy())) for g in adj_test]).to(device) for v in range(args.num_views)]
            batchSize_info_test = [len(adj_test), adj_test[0].shape[-1]]

        # Forward pass for the mini-batch. Use the mean for stable evaluation.
        # The StagedSupervisedVAE returns node embeddings, graph_mean, graph_log_std
            _, mean_test, _, _ = model(dgl_views_test, features_dgl_test, batchSize_info_test)
        
        # Get logits from the decoder
            test_logits = model.decoder(mean_test)
        
        # Collect predictions and labels from this batch
            all_preds.append(torch.sigmoid(test_logits).cpu())
            all_labels.append(torch.tensor(labels_test))

# Concatenate results from all mini-batches
    all_preds = torch.cat(all_preds).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Epoch: {epoch+1:03d} | Avg Loss: {avg_total_loss:.4f} | Class Loss: {avg_class_loss:.4f} | "
        f"KL Loss: {avg_kl_loss:.4f} | Beta: {kl_beta:.3f} | Test AUC: {auc:.4f}")
    logging.info(f"Epoch: {epoch+1:03d},Loss:{avg_total_loss:.4f},Class Loss:{avg_class_loss:.4f},KL Loss:{avg_kl_loss:.4f},Test AUC:{auc:.4f}")

# ========================================================================
#
#                           STAGE 2: DIFFUSION CLASSIFIER
#
# ========================================================================
print("\n--- Stage 1 Pre-training Finished. ---")
print("--- Generating pre-trained embeddings for Stage 2 ---")
if 'optimizer' in locals():
    del optimizer
torch.cuda.empty_cache()

torch._dynamo.config.capture_scalar_outputs = True
if args.task == "graphClassification":
    print("\n--- Starting Stage 2: Multi-Label Classification with Graph-UNet Diffusion ---")
    logging.info("\n--- Starting Stage 2: Multi-Label Classification with Graph-UNet Diffusion ---")

    # =================== FIX 1: CHANGE THE IMPORT ===================
    from torch.cuda.amp import GradScaler # Keep GradScaler
    # torch.amp is now used directly, no need for a specific autocast import
    # ================================================================
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # --- 1. Prepare Data ---
    train_labels = torch.tensor(np.array(list_graphs.labels), dtype=torch.float32).to(device)
    test_labels = torch.tensor(np.array(list_test_graphs.labels), dtype=torch.float32).to(device)

    # --- 2. Generate Graph Embeddings (in batches to conserve memory) ---
    print("Generating embeddings for train/test sets (using batched inference for memory efficiency)...")
    logging.info("Generating embeddings for train/test sets (using batched inference for memory efficiency)...")
    model.eval()

    def get_embeddings_batched(dataset, model, batch_size, device, num_views, feature_size, max_nodes):
        embeddings_list_cpu = []
        with torch.no_grad():
            for i in range(0, len(dataset.adj_s), batch_size):
                end = min(i + batch_size, len(dataset.adj_s))
                adj_batch = dataset.adj_s[i:end]
                features_batch = dataset.x_s[i:end]
                dgl_views_batch = []
                for v in range(num_views):
                    scipy_graphs = [sp.csr_matrix(g[v].cpu().numpy()) for g in adj_batch]
                    dgl_graphs = [dgl.from_scipy(sg) for sg in scipy_graphs]
                    dgl_views_batch.append(dgl.batch(dgl_graphs).to(device))
                features_tensor_batch = torch.stack(features_batch).reshape(-1, feature_size).to(device)
                batch_size_info = [len(adj_batch), max_nodes]
                _, graph_level_mean, _ = model.encoder(dgl_views_batch, features_tensor_batch, batch_size_info)
                batch_embeddings = graph_level_mean
                embeddings_list_cpu.append(batch_embeddings.cpu())
                del dgl_views_batch, features_tensor_batch, batch_embeddings, graph_level_mean
        final_embeddings = torch.cat(embeddings_list_cpu, dim=0).to(device)
        return final_embeddings

    gen_batch_size = 16
    train_embeddings = get_embeddings_batched(list_graphs, model, gen_batch_size, device, args.num_views, list_graphs.feature_size, list_graphs.max_num_nodes)
    test_embeddings = get_embeddings_batched(list_test_graphs, model, gen_batch_size, device, args.num_views, list_test_graphs.feature_size, list_test_graphs.max_num_nodes)
    train_dgl_graph_list = [dgl.from_scipy(sp.csr_matrix(g[0].cpu().numpy())) for g in list_graphs.adj_s]
    test_dgl_graphs = [dgl.from_scipy(sp.csr_matrix(g[0].cpu().numpy())) for g in list_test_graphs.adj_s]
    test_dgl_graph_batched = dgl.batch(test_dgl_graphs).to(device)

    print(f"Generated {train_embeddings.shape[0]} training embeddings (dim={train_embeddings.shape[1]}) and {test_embeddings.shape[0]} test embeddings.")
    logging.info(f"Generated {train_embeddings.shape[0]} training embeddings (dim={train_embeddings.shape[1]}) and {test_embeddings.shape[0]} test embeddings.")
    
    # --- SANITY CHECKS with Simple Classifiers ---
    X_train = train_embeddings.cpu().numpy()
    y_train = train_labels.cpu().numpy().ravel()
    X_test = test_embeddings.cpu().numpy()
    y_test = test_labels.cpu().numpy().ravel()

    print("\n--- Running Sanity Checks on VAE Embeddings ---")
    logging.info("\n--- Running Sanity Checks on VAE Embeddings ---")
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score_sklearn(y_test, y_pred_proba_lr)
    print(f"--- Logistic Regression AUC: {lr_auc:.4f} ---")
    logging.info(f"--- Logistic Regression AUC: {lr_auc:.4f} ---")
    svm_pipeline = make_pipeline(StandardScaler(), SVC(class_weight='balanced', probability=True, random_state=42))
    svm_pipeline.fit(X_train, y_train)
    y_pred_proba_svm = svm_pipeline.predict_proba(X_test)[:, 1]
    svm_auc = roc_auc_score_sklearn(y_test, y_pred_proba_svm)
    print(f"--- SVM Classifier AUC: {svm_auc:.4f} ---")
    logging.info(f"--- SVM Classifier AUC: {svm_auc:.4f} ---")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score_sklearn(y_test, y_pred_proba_rf)
    print(f"--- Random Forest AUC: {rf_auc:.4f} ---")
    logging.info(f"--- Random Forest AUC: {rf_auc:.4f} ---")

    # --- 3. Define and Train the Diffusion Classifier ---
    ddm_config = { "num_hidden": 256, "num_layers": 4, "nhead": 4, "activation": 'gelu', 
                   "feat_drop": 0.1, "attn_drop": 0.1, "norm": 'layernorm', "T": 500, "beta_schedule": 'cosine' }
    
    diffusion_classifier_embedding_dim = train_embeddings.shape[1]
    diffusion_classifier = LabelDiffusionClassifier(graph_embedding_dim=diffusion_classifier_embedding_dim, DDM_config=ddm_config).to(device)
    print("Skipping torch.compile() due to C++ compilation error with DGL.")

    optimizer = torch.optim.Adam(diffusion_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler()
    
    def evaluate_diffusion_model(model, test_graph, test_embeds, test_lbls):
        model.eval()
        with torch.no_grad():
            # =================== FIX 2: UPDATE AUTOCAST CALL ===================
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # ===================================================================
                predicted_logits = model.sample(test_graph, test_embeds)
                predicted_logits = predicted_logits.float()

        device = predicted_logits.device
        preds = (predicted_logits > 0.5)
        target_labels = test_lbls.long()
        acc_metric = Accuracy(task="binary").to(device)
        acc = acc_metric(preds, target_labels).item()
        f1_metric = F1Score(task="binary").to(device)
        f1 = f1_metric(preds, target_labels).item()
        try:
            auroc_metric = AUROC(task="binary").to(device)
            auc_val = auroc_metric(predicted_logits, target_labels).item()
            macro_auc = auc_val
            micro_auc = auc_val
        except (ValueError, IndexError):
            macro_auc, micro_auc = -1.0, -1.0
        return acc, f1, macro_auc, micro_auc

    print("\n--- Training the Diffusion Classifier ---")
    logging.info("\n--- Training the Diffusion Classifier ---")
    num_diffusion_epochs = 800
    batch_size = 16
    eval_every_epochs = 1 

    train_indices = list(range(len(list_graphs.adj_s)))

    for epoch in range(num_diffusion_epochs):
        diffusion_classifier.train() 
        random.shuffle(train_indices)
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            if not batch_indices: continue
            graphs_for_batch = [train_dgl_graph_list[j] for j in batch_indices]
            batch_dgl_graph = dgl.batch(graphs_for_batch).to(device)
            batch_embeddings = train_embeddings[batch_indices]
            batch_labels = train_labels[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            
            # =================== FIX 3: UPDATE AUTOCAST CALL ===================
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # ===================================================================
                loss, _ = diffusion_classifier(batch_dgl_graph, batch_embeddings, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % eval_every_epochs == 0 or (epoch + 1) == num_diffusion_epochs:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            test_acc, test_f1, test_macro_auc, test_micro_auc = evaluate_diffusion_model(
                model=diffusion_classifier,
                test_graph=test_dgl_graph_batched,
                test_embeds=test_embeddings,
                test_lbls=test_labels
            )
            log_message = (f"Epoch {epoch+1}/{num_diffusion_epochs} | "
                           f"Loss: {avg_loss:.4f} | "
                           f"Test Acc: {test_acc:.4f} | "
                           f"Test F1: {test_f1:.4f} | "
                           f"Test AUC: {test_macro_auc:.4f}")
            print(log_message)
            logging.info(log_message)

    print("\n--- Final Diffusion Model Evaluation ---")
    logging.info("\n--- Final Diffusion Model Evaluation ---")
    print("Training finished.")
    logging.info("Training finished.")