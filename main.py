#
# main.py (Modified)
#
import logging
import plotter
import torch.nn.functional as F
import argparse
from model import *
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl
from label_diffusion import LabelDiffusionClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
from torchmetrics.classification import Accuracy, F1Score, AUROC
import torch._dynamo
import torch.amp

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

subgraphSize = None
keepThebest = False

parser = argparse.ArgumentParser(description='Kernel VGAE')

parser.add_argument('-e', dest="epoch_number", default=10000, help="Number of Epochs to train the model", type=int)
parser.add_argument('-v', dest="Vis_step", default=1000, help="at every Vis_step 'minibatch' the plots will be updated")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.0003, help="model learning rate")
parser.add_argument('-dataset', dest="dataset", default="Multi",
                    help="possible choices are:   wheel_graph,PTC, FIRSTMM_DB, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")  # citeceer: ego; DD:protein
parser.add_argument('-graphEmDim', dest="graphEmDim", default=1024, help="the dimention of graph Embeding LAyer; z")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None,
                    help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature", default=True, help="either use features or identity matrix")
parser.add_argument('-PATH', dest="PATH", default="model",
                    help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder", default="FC", help="the decoder type, FC is only option in this rep")
parser.add_argument('-encoder', dest="encoder_type", default="AvePool",
                    help="the encoder: only option in this rep is 'AvePool'")  # only option in this rep is "AvePool"
parser.add_argument('-batchSize', dest="batchSize", default=200,
                    help="the size of each batch; the number of graphs is the mini batch")
parser.add_argument('-UseGPU', dest="UseGPU", default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model", default="GraphVAE-MM",
                    help="KernelAugmentedWithTotalNumberOfTriangles and kipf is the only option in this rep; NOTE KernelAugmentedWithTotalNumberOfTriangles=GraphVAE-MM and kipf=GraphVAE")
parser.add_argument('-device', dest="device", default="cuda:2", help="Which device should be used")
#parser.add_argument('-task', dest="task", default="graphGeneration", help="only option in this rep is graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering", default=False, help="use bfs for graph permutations", type=bool) ### MODIFICATION 1: Changed default to False
parser.add_argument('-directed', dest="directed", default=True, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta", default=None, help="beta coefiicieny", type=float)
parser.add_argument('-plot_testGraphs', dest="plot_testGraphs", default=True, help="shall the test set be printed",
                    type=float)
parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution" , default=False, help="if you want to comapre the 50%50 subset of dataset comparision?!", type=bool)

parser.add_argument('-num_views', dest="num_views", default=2, help="Number of views in the multi-view graph dataset", type=int)
parser.add_argument('-task', dest="task", default="graphClassification", help="options: graphGeneration, graphClassification")


args = parser.parse_args()
ideal_Evalaution = args.ideal_Evalaution
encoder_type = args.encoder_type
graphEmDim = args.graphEmDim
visulizer_step = args.Vis_step
redraw = args.redraw
device = args.device
task = args.task
plot_testGraphs = args.plot_testGraphs
directed = args.directed
epoch_number = args.epoch_number
lr = args.lr
decoder_type = args.decoder
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
mini_batch_size = args.batchSize
use_gpu = args.UseGPU
use_feature = args.use_feature

graph_save_path = args.graph_save_path
graph_save_path = args.graph_save_path

num_views = args.num_views#new add

if graph_save_path == None:
    graph_save_path = "MMD_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + args.model + "BFS" + str(
        args.bfsOrdering) + str(args.epoch_number) + str(time.time()) + "/"
from pathlib import Path

Path(graph_save_path).mkdir(parents=True, exist_ok=True)

# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=graph_save_path + 'log.log', filemode='w', level=logging.INFO)

# **********************************************************************
# setting; general setting and hyper-parameters for each dataset
print("KernelVGAE SETING: " + str(args))
logging.info("KernelVGAE SETING: " + str(args))
PATH = args.PATH  # the dir to save the with the best performance on validation data

kernl_type = []
#---------------------------------------------------------------------
if args.model == "KernelAugmentedWithTotalNumberOfTriangles" or args.model=="GraphVAE-MM":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]
    if dataset=="mnist":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset=="zinc":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset == "large_grid":
        step_num = 5 # s in s-step transition
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 100]
    elif dataset == "ogbg-molbbbp":
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 0, 0, 1, 40, 1500]
        # -----------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
        step_num = 5
    elif dataset == "IMDBBINARY":
        alpha = [ 1, 1, 1, 1, 1, 1, 2, 50]
        step_num = 5
    elif dataset == "QM9":
        step_num = 2
        alpha = [ 1, 1, 1, 1, 1, 20, 200]
    elif dataset == "PTC":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    elif dataset =="MUTAG":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 60]
    elif dataset =="PVGAErandomGraphs":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1]
    elif dataset == "FIRSTMM_DB":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 100]
    elif dataset == "DD":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1000]
    elif dataset == "grid":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "lobster":
        step_num = 5
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]  # degree
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 2000]  # degree
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]
        # -------------------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 2000]
    elif dataset == "wheel_graph":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 3000000, 20000 * 50000]
    elif dataset == "triangular_grid":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "tree":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]

    elif dataset == "Multi":
        print(f"Setting simple hyperparameters for {dataset}.")
        # Disable kernel statistics for the initial run
        kernl_type = [] 
        step_num = 0
        # Set weights only for the last two loss terms:
        # 1. Binary Cross-Entropy (reconstruction loss)
        # 2. KL Divergence (regularization)
        alpha = [10.0, 1.0] # [weight_BCE, weight_KL]
#---------------------------------------------------------------------

elif args.model == "kipf" or args.model == "graphVAE":
    alpha = [1, 1]
    step_num = 0

AutoEncoder = False

if AutoEncoder == True:
    alpha[-1] = 0

if args.beta != None:
    alpha[-1] = args.beta

print("kernl_type:" + str(kernl_type))
print("alpha: " + str(alpha) + " num_step:" + str(step_num))

logging.info("kernl_type:" + str(kernl_type))
logging.info("alpha: " + str(alpha) + " num_step:" + str(step_num))

  # with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :" + str(device))

# setting the plots legend
functions = ["Accuracy", "loss"]
if args.model == "kernel" or args.model == "KernelAugmentedWithTotalNumberOfTriangles" or args.model == "GraphVAE-MM":
    functions.extend(["Kernel" + str(i) for i in range(step_num)])
    functions.extend(kernl_type[1:])

if args.model == "TrianglesOfEachNode":
    functions.extend(kernl_type)

if args.model == "ThreeStepPath":
    functions.extend(kernl_type)

if args.model == "TotalNumberOfTriangles":
    functions.extend(kernl_type)

functions.append("Binary_Cross_Entropy")
functions.append("KL-D")

# ========================================================================


pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log", functions=functions)

synthesis_graphs = {"wheel_graph", "star", "triangular_grid", "DD", "ogbg-molbbbp", "grid", "small_lobster",
                    "small_grid", "community", "lobster", "ego", "one_grid", "IMDBBINARY", ""}


class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num, outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InLatent_dim * outNode_num)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0, 2, 1), inTensor)

        return activation(Z)


class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num, InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InNode_num * OutLatentDim)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1))

        return activation(Z)


# ============================================================================

def test_(number_of_samples, model, graph_size, path_to_save_g, remove_self=True, save_graphs=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_array(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(k) + str(g_size) + str(j) + dataset
            k += 1
            # plot and save the generated graph
            # plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))

            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            if save_graphs:
                plotter.plotG(G, "generated" + dataset, file_name=f_name + "_ConnectedComponnents")
    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + args.model + "_" + task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list


def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name=None, onlyTheBigestConCom = True):
    generated_graphs = test_(1, model, [x.shape[0] for x in test_list_adj], graph_save_path, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
    if (onlyTheBigestConCom==False):
        if Save_generated:
            np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                    allow_pickle=True)


            logging.info(mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if
                        not nx.is_empty(G)]

    statistic_   = mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj], diam=True)
    # if writeThem_in!=None:
    #     with open(writeThem_in+'MMD.log', 'w') as f:
    #         f.write(statistic_)
    logging.info(statistic_)
    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
        np.save(graph_save_path + 'Single_comp_generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for G in test_list_adj]
        np.save(graph_save_path + 'testGraphs_adj_.npy', graphs_to_writeOnDisk, allow_pickle=True)
    return  statistic_

def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    subgraphs = []
    target_kelrnel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes != None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model != None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kelrnel_val, subgraphs


# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


# def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, alpha,
#                  reconstructed_adj_logit, pos_wight, norm):
#     loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
#                                                                        targert_adj.float(), pos_weight=pos_wight)

#     norm = mean.shape[0] * mean.shape[1]
#     kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

#     acc = (reconstructed_adj.round() == targert_adj).sum() / float(
#         reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
#     kernel_diff = 0
#     each_kernel_loss = []
#     log_sigma_values = []
#     for i in range(len(target_kernel_val)):
#         log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
#         log_sigma = softclip(log_sigma, -6)
#         log_sigma_values.append(log_sigma.detach().cpu().item())
#         step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
#         each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
#         kernel_diff += step_loss * alpha[i]

#     kernel_diff += loss * alpha[-2]
#     kernel_diff += kl * alpha[-1]
#     each_kernel_loss.append((loss * alpha[-2]).item())
#     each_kernel_loss.append((kl * alpha[-1]).item())
#     return kl, loss, acc, kernel_diff, each_kernel_loss,log_sigma_values

# def OptimizerVAE(reconstructed_adj_logit_views, reconstructed_kernel_val_views, 
#                  target_adj_views, target_kernel_val_views, 
#                  log_std, mean, alpha, pos_wight, norm):#new one
    
#     total_bce_loss = 0
#     total_kernel_loss = 0
#     num_views = reconstructed_adj_logit_views.shape[1]

#     # --- Micro Loss (Reconstruction) ---
#     for v in range(num_views):
#         recon_logit_v = reconstructed_adj_logit_views[:, v, :, :]
#         target_adj_v = target_adj_views[:, v, :, :]
#         total_bce_loss += F.binary_cross_entropy_with_logits(recon_logit_v.float(),
#                                                               target_adj_v.float(), pos_weight=pos_wight)
    
#     bce_loss = total_bce_loss / num_views # Average loss across views

#     # --- Macro Loss (Kernel Statistics) ---
#     each_kernel_loss = []
#     if target_kernel_val_views and reconstructed_kernel_val_views:
#         # Assuming target_kernel_val_views is structured as: [[view1_stats], [view2_stats], ...]
#         for v in range(num_views):
#             target_kernels_v = target_kernel_val_views[v]
#             recon_kernels_v = reconstructed_kernel_val_views[v]
#             for i in range(len(target_kernels_v)):
#                 log_sigma = ((recon_kernels_v[i] - target_kernels_v[i]) ** 2).mean().sqrt().log()
#                 log_sigma = softclip(log_sigma, -6)
#                 step_loss = log_guss(target_kernels_v[i], log_sigma, recon_kernels_v[i]).mean()
#                 total_kernel_loss += step_loss * alpha[i]
#                 each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
    
#     kernel_loss = total_kernel_loss # Sum of kernel losses across all views and stats

#     # --- KL Divergence ---
#     kl = (1 / (mean.shape[0] * mean.shape[1])) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

#     # --- Final Loss Combination ---
#     final_loss = (bce_loss * alpha[-2]) + kernel_loss + (kl * alpha[-1])
    
#     # ... (accuracy calculation can be tricky for multi-view, maybe avg it)
#     acc = (torch.sigmoid(reconstructed_adj_logit_views).round() == target_adj_views).sum() / float(
#         target_adj_views.nelement())

#     return kl, bce_loss, acc, final_loss, each_kernel_loss, [] # return empty log_sigma for now

# In main.py

def OptimizerVAE(reconstructed_adj_logit_views, reconstructed_kernel_val_views,
                 target_adj_views, target_kernel_val_views,
                 log_std, mean, alpha, pos_wight, norm, beta):

    num_views = reconstructed_adj_logit_views.shape[1]

    max_nodes_in_batch = reconstructed_adj_logit_views.shape[-1]
    target_adj_views_sliced = target_adj_views[:, :, :max_nodes_in_batch, :max_nodes_in_batch]

    # --- 1. Reconstruction Loss (BCE) ---
    # This must be a positive value. We calculate it for each view and average.
    total_bce_loss = 0
    for v in range(num_views):
        recon_logit_v = reconstructed_adj_logit_views[:, v, :, :]
        target_adj_v = target_adj_views_sliced[:, v, :, :]
        # F.binary_cross_entropy_with_logits returns a positive loss value.
        total_bce_loss += F.binary_cross_entropy_with_logits(
            recon_logit_v.float(),
            target_adj_v.float(),
            pos_weight=pos_wight
        )
    bce_loss = total_bce_loss / num_views # Average across views.

    # --- 2. Kernel Loss ---
    # This part is for your specific model variant. It should also be positive.
    # The original implementation using log_guss is correct.
    each_kernel_loss = []
    kernel_loss = 0
    if target_kernel_val_views and reconstructed_kernel_val_views and kernl_type:
        # Loop through each view
        for v in range(num_views):
            target_kernels_v = target_kernel_val_views[v]
            recon_kernels_v = reconstructed_kernel_val_views[v]
            # Loop through each kernel statistic for that view
            for i in range(len(target_kernels_v)):
                log_sigma = ((recon_kernels_v[i] - target_kernels_v[i]) ** 2).mean().sqrt().log()
                log_sigma = softclip(log_sigma, -6)
                step_loss = log_guss(target_kernels_v[i], log_sigma, recon_kernels_v[i]).mean()
                # Use the alpha weights for the kernel terms
                weighted_step_loss = step_loss * alpha[i]
                kernel_loss += weighted_step_loss
                each_kernel_loss.append(weighted_step_loss.cpu().detach().numpy())

    # --- 3. KL Divergence ---
    # This is the standard KL divergence formula for a VAE, which is positive.
    kl_norm_factor = mean.shape[0] * mean.shape[1]
    kl_loss = (1 / kl_norm_factor) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    # --- 4. Final Loss Combination ---
    # All components (bce_loss, kernel_loss, kl_loss) are positive.
    # We apply their respective weights.
    # alpha[-2] is the weight for BCE loss (e.g., 10.0)
    # beta is the dynamic weight for KL loss (0 -> 1)
    final_loss = (bce_loss * alpha[-2]) + kernel_loss + (beta * kl_loss)
    acc = (torch.sigmoid(reconstructed_adj_logit_views).round() == target_adj_views_sliced).sum() / float(
        target_adj_views_sliced.nelement())

    return kl_loss, bce_loss, acc, final_loss, each_kernel_loss, []


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# test_(5, "results/multiple graph/cora/model" , [x**2 for x in range(5,10)])


# load the data
print("--- Loading Raw Data ---")
list_adj, list_x, list_label = list_graph_loader(args.dataset, return_labels=True)  # , _max_list_size=80)
# list_adj = list_adj[:400]
# list_x = list_x[:400]
# list_label = list_label[:400]

### MODIFICATION 1: Remove canonical ordering by commenting out the BFS call.
# The GCN-based model should be permutation-invariant by design, so this step is not strictly necessary.
# if args.bfsOrdering == True:
#     print("Applying BFS ordering to the dataset...")
#     list_adj = BFS(list_adj)
# else:
#     print("Skipping BFS ordering.")
### END MODIFICATION 1

# list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True, _max_list_size=80)

# list_adj, _ = permute(list_adj, None)
self_for_none = True
if (decoder_type) in ("FCdecoder"):  # ,"FC_InnerDOTdecoder"
    self_for_none = True

if len(list_adj) == 1:
    test_list_adj = list_adj.copy()
    list_graphs = Datasets(list_adj, self_for_none, list_x, None)
else:
    max_size = None
    # list_label = None
    list_adj, test_list_adj, list_x_train, list_x_test, _ ,list_label_test= data_split(list_adj, list_x,list_label)
    val_adj = list_adj[:int(len(test_list_adj))]
    list_graphs = Datasets(list_adj, self_for_none, list_x_train, list_label, Max_num=max_size,
                           set_diag_of_isol_Zer=False)
    list_test_graphs = Datasets(test_list_adj, self_for_none, list_x_test, list_label_test, Max_num=list_graphs.max_num_nodes,
                           set_diag_of_isol_Zer=False)
    if plot_testGraphs:
        print("printing the test set...")
        # for i, G in enumerate(test_list_adj):
        #     G = nx.from_numpy_matrix(G.toarray())
        #     plotter.plotG(G, graph_save_path+"_test_graph" + str(i))

print("#------------------------------------------------------")
if ideal_Evalaution:
    fifty_fifty_dataset = list_adj + test_list_adj

    fifty_fifty_dataset = [nx.from_numpy_array(graph.toarray()) for graph in fifty_fifty_dataset]
    random.shuffle(fifty_fifty_dataset)
    print("50%50 Evalaution of dataset")
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))

    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in fifty_fifty_dataset]
    np.save(graph_save_path+dataset+'_dataset.npy', graphs_to_writeOnDisk, allow_pickle=True)
print("#------------------------------------------------------")

print("Processing training and testing data to determine feature sizes...")
list_graphs.processALL(self_for_none=self_for_none)
list_test_graphs.processALL(self_for_none=self_for_none)

SubGraphNodeNum = subgraphSize if subgraphSize != None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size  # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes

print(f"--- Data Processed. Determined input feature dimension: {in_feature_dim} ---")

degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum,1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
bin_width = torch.tensor([[1] for x in range(0, SubGraphNodeNum, 1)])

kernel_model = kernel(device=device, kernel_type=kernl_type, step_num=step_num,
                      bin_width=bin_width, bin_center=bin_center, degree_bin_center=degree_center,
                      degree_bin_width=degree_width)

if encoder_type == "AvePool":
    # encoder = AveEncoder(in_feature_dim, [256], graphEmDim)
    encoder = MultiViewAveEncoder(in_feature_dim=in_feature_dim,
        num_views=args.num_views,
        hiddenLayers=[256, 256],
        GraphLatntDim=graphEmDim) #update to fit multi-view
else:
    print("requested encoder is not implemented")
    exit(1)

if decoder_type == "FC":
    # decoder = GraphTransformerDecoder_FC(graphEmDim, 256, nodeNum, directed)
    # decoder = GraphTransformerDecoder_FC(input_dim=graphEmDim,
    #     lambdaDim=None, # This parameter seems unused in your code
    #     SubGraphNodeNum=nodeNum,
    #     num_views=args.num_views,
    #     directed=False)#update to fit multi-view

    decoder = NodeLevelInnerProductDecoder(
    in_node_dim=graphEmDim, # The dimension of each node's latent vector
    num_views=args.num_views
)
else:
    print("requested decoder is not implemented")
    exit(1)

model = kernelGVAE(ker=kernel_model, 
    encoder=encoder, 
    decoder=decoder, 
    AutoEncoder=AutoEncoder,
    graphEmDim=graphEmDim,
    num_views=args.num_views) # <<< Pass num_views)  # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)

# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
# ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

list_graphs.shuffle()
start = timeit.default_timer()
# Parameters
step = 0
swith = False
print(model)
logging.info(model.__str__())
min_loss = float('inf')

# --- NEW: KL Annealing Parameters ---
kl_beta = 0.0
# Increase beta over the first 500 training steps
kl_anneal_epochs = 25 # Let's anneal over the first 25 epochs.

# Calculate the number of batches/steps per epoch
steps_per_epoch = max(1, len(list_graphs.list_adjs) // mini_batch_size)

# Calculate the total number of steps for annealing
kl_anneal_steps = steps_per_epoch * kl_anneal_epochs

# Ensure we don't divide by zero if the period is very short
if kl_anneal_steps == 0:
    kl_anneal_steps = 1

kl_anneal_rate = 1.0 / kl_anneal_steps
kl_anneal_target = 1.0 # The final value for beta
print(f"KL Annealing will occur over the first {kl_anneal_epochs} epochs ({kl_anneal_steps} steps).")
# ------------------------------------

# ------------------------------------

# if (subgraphSize == None):
#     list_graphs.processALL(self_for_none=self_for_none)
#     adj_list = list_graphs.get_adj_list()
#     graphFeatures, _ = get_subGraph_features(adj_list, None, kernel_model)
#     list_graphs.set_features(graphFeatures)

# 50%50 Evaluation

load_model = False
if load_model == True:  # I used this in line code to load a model #TODO: fix it
    # ========================================
    model_dir = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_DD_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651364417.4785793/"
    model.load_state_dict(torch.load(model_dir + "model_9999_3"))
    # EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )

# model_dir1 = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/"
# model.load_state_dict(torch.load(model_dir1+"model_9999_3"))
# EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )


for epoch in range(epoch_number):

    list_graphs.shuffle()
    batch = 0
    for iter in range(0, max(int(len(list_graphs.list_adjs) / mini_batch_size), 1) * mini_batch_size, mini_batch_size):
        from_ = iter
        to_ = mini_batch_size * (batch + 1)
        # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
        #     from_ = iter
        #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)
        
        
        
        # if subgraphSize == None:
        #     org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none,
        #                                                                                       bfs=subgraphSize)
        # else:
        #     org_adj, x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)
        org_adj, x_s, node_num, _, target_kelrnel_val = list_graphs.get__(
            from_, to_, self_for_none, bfs=None # Use bfs=None to get pre-processed data
        )
        subgraphs_target_tensor = torch.stack(org_adj).to(device)

        if (type(decoder)) in [GraphTransformerDecoder_FC]:  #
            node_num = len(node_num) * [list_graphs.max_num_nodes]

        x_s_tensor = torch.stack(x_s).to(device)

        model.train()

        total_nodes = x_s_tensor.shape[0] * x_s_tensor.shape[1]
        feature_dim = x_s_tensor.shape[2]
        features_for_dgl = x_s_tensor.reshape(total_nodes, feature_dim).to(device)
        

        # if subgraphSize == None:
        #     _, subgraphs = get_subGraph_features(org_adj, None, None)
        # else:
        #     target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[-1]]
        batchSize_for_encoder = [len(org_adj), org_adj[0].shape[1]] # (batch_size, num_nodes)


        dgl_graphs_per_view = []
        for v in range(args.num_views):
            view_graphs_in_batch = [dgl.from_scipy(sp.csr_matrix(g[v].cpu().numpy())) for g in org_adj]
            dgl_graphs_per_view.append(dgl.batch(view_graphs_in_batch).to(device))

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        # [graph.setdiag(1) for graph in org_adj]
        # org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        # org_adj_dgl = dgl.batch(org_adj_dgl).to(device)

        # pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
        pos_wight = torch.true_divide(
            subgraphs_target_tensor.numel() - subgraphs_target_tensor.sum(),
            subgraphs_target_tensor.sum()
        ).clamp(min=1.0, max=50.0)

        # reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
        #     org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
        # kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss,log_sigma_values = OptimizerVAE(reconstructed_adj,
        #                                                                                 generated_kernel_val,
        #                                                                                 subgraphs.to(device),
        #                                                                                 [val.to(device) for val in
        #                                                                                  target_kelrnel_val],
        #                                                                                 post_log_std, post_mean, alpha,
        #                                                                                 reconstructed_adj_logit,
        #                                                                                 pos_wight, 2)

        # loss = kernel_cost

        if kl_beta < kl_anneal_target:
            kl_beta = min(kl_anneal_target, kl_beta + kl_anneal_rate)

        reconstructed_adj_views, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
            dgl_graphs_per_view, # Pass the list of DGL graphs
            features_for_dgl,
            batchSize,
            subgraphs_indexes = None
        )
        
        #Call the correct OptimizerVAE with multi-view arguments
        kl_loss, reconstruction_loss, acc, total_loss, each_kernel_loss, log_sigma_values = OptimizerVAE(
            reconstructed_adj_logit,           # Logits for all views
            generated_kernel_val,              # Kernel stats for all views
            subgraphs_target_tensor,           # Target adjacency tensors for all views
            target_kelrnel_val,                # Target kernel stats for all views
            post_log_std,
            post_mean,
            alpha,
            pos_wight,
            norm=2, # Your original norm value
            beta=kl_beta
        )
        
        loss = total_loss#new

        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss], tmp,
                        redraw=redraw)  # ["Accuracy", "loss", "AUC"])

        step += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        if keepThebest and min_loss > loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (step + 1) % visulizer_step == 0 or epoch_number==epoch+1:
            model.eval()
            pltr.redraw()
            if True:
                dir_generated_in_train = "generated_graph_train/"
                if not os.path.isdir(dir_generated_in_train):
                    os.makedirs(dir_generated_in_train)

                rnd_indx = random.randint(0, len(node_num) - 1)
                
                # Get the first view [:, 0, :, :] for all graphs in the batch
                reconstructed_first_view = torch.sigmoid(reconstructed_adj_logit)[:, 0, :, :]

                # Now select a random sample from that view
                sample_graph = reconstructed_first_view[rnd_indx].cpu().detach().numpy()
                sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
                sample_graph[sample_graph >= 0.5] = 1
                sample_graph[sample_graph < 0.5] = 0

                G = nx.from_numpy_array(sample_graph)
                plotter.plotG(G, "generated_" + dataset + "_view0",
                              file_name=graph_save_path + "generatedSample_At_epoch" + str(epoch))
                
                print("reconstructed graph vs Validation:")
                logging.info("reconstructed graph vs Validation:")

                # The rest of this evaluation logic also needs to be updated
                reconstructed_adj_for_eval = reconstructed_first_view.cpu().detach().numpy()
                reconstructed_adj_for_eval[reconstructed_adj_for_eval >= 0.5] = 1
                reconstructed_adj_for_eval[reconstructed_adj_for_eval < 0.5] = 0
                
                reconstructed_graphs = [nx.from_numpy_array(reconstructed_adj_for_eval[i]) for i in range(reconstructed_adj_for_eval.shape[0])]
                reconstructed_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                     reconstructed_graphs if not nx.is_empty(G)]

                # The target set also needs to be multi-view aware
                target_set = [nx.from_numpy_array(val_adj[i][0].toarray()) for i in range(len(val_adj))] # Using first view of validation set
                target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                            not nx.is_empty(G)]

                reconstruc_MMD_loss = mmd_eval(reconstructed_graphs, target_set[:len(reconstructed_graphs)], diam=True)
                logging.info(reconstruc_MMD_loss)

            #todo: instead of printing diffrent level of logging shoud be used
            model.eval()
            # if task == "graphGeneration":
            #     # print("generated vs Validation:")
            #     mmd_res= EvalTwoSet(model, val_adj[:1000], graph_save_path, Save_generated=True, _f_name=epoch)
            #     with open(graph_save_path + '_MMD.log', 'a') as f:
            #             f.write(str(step)+" @ loss @ , "+str(loss.item())+" , @ Reconstruction @ , "+reconstruc_MMD_loss+" , @ Val @ , " +mmd_res+"\n")

            #     if ((step + 1) % visulizer_step * 2):
            #         torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
            stop = timeit.default_timer()
            # print("trainning time at this epoch:", str(stop - start))
            model.train()
            # if reconstruction_loss.item()<0.051276 and not swith:
            #     alpha[-1] *=2
            #     swith = True
        k_loss_str = ""
        for indx, l in enumerate(each_kernel_loss):
            k_loss_str += functions[indx + 2] + ":"
            k_loss_str += str(l) + ".   "

        print(
            "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Beta: {:.3f} | accu: {:03f}".format(
                epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), kl_beta, acc), k_loss_str)
        logging.info(
            "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), acc) + " " + str(k_loss_str))
        # print(log_sigma_values)
        log_std = ""
        for indx, l in enumerate(log_sigma_values):
            log_std += "log_std " + functions[indx + 2] + ":"
            log_std += str(l) + ".   "
        print(log_std)
        logging.info(log_std)
        batch += 1
        # scheduler.step()
model.eval()
torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))

stop = timeit.default_timer()
print("trainning time:", str(stop - start))
logging.info("trainning time: " + str(stop - start))
# save the train loss for comparing the convergence
import json

file_name = graph_save_path + "_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + args.model + "_elbo_loss.txt"

with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2]) + np.array(pltr.values_train[-1])), fp)

# with open(file_name + "/_CrossEntropyLoss.txt", "w") as fp:
#     json.dump(list(np.array(pltr.values_train[-2])), fp)
#
# with open(file_name + "/_train_loss.txt", "w") as fp:
#     json.dump(pltr.values_train[1], fp)

# save the log plot on the current directory
pltr.save_plot(graph_save_path + "KernelVGAE_log_plot")

if task == "graphGeneration":
    EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name="final_eval")

# ... (The old classification code is removed for clarity) ...

### MODIFICATION 2 & 3: This section is heavily modified to pass the full graph view
### to the diffusion model and to handle the new logic in LabelDiffusionClassifier.
# In main.py
torch._dynamo.config.capture_scalar_outputs = True
if args.task == "graphClassification":
    print("\n--- Starting Stage 2: Multi-Label Classification with Graph-UNet Diffusion ---")
    logging.info("\n--- Starting Stage 2: Multi-Label Classification with Graph-UNet Diffusion ---")

    import torch.amp
    from label_diffusion import LabelDiffusionClassifier
    from torchmetrics.classification import Accuracy, F1Score, AUROC
    from torch.cuda.amp import GradScaler, autocast

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # --- 1. Prepare Data ---
    train_labels = torch.tensor(np.array(list_graphs.labels), dtype=torch.float32).to(device)
    test_labels = torch.tensor(np.array(list_test_graphs.labels), dtype=torch.float32).to(device)

    # --- 2. Generate Graph Embeddings (This is a one-time cost at startup) ---
    print("Generating embeddings and preparing DGL graphs for train and test sets...")
    logging.info("Generating embeddings and preparing DGL graphs for train and test sets...")
    model.eval()
    with torch.no_grad():
        # --- Get Train Data ---
        train_dgl_views = []
        for v in range(args.num_views):
            scipy_graphs = [sp.csr_matrix(g[v].cpu().numpy()) for g in list_graphs.adj_s]
            dgl_graphs = [dgl.from_scipy(sg) for sg in scipy_graphs]
            train_dgl_views.append(dgl.batch(dgl_graphs).to(device))
        train_features = torch.stack(list_graphs.x_s).reshape(-1, list_graphs.feature_size).to(device)
        train_batch_size = [len(list_graphs.adj_s), list_graphs.max_num_nodes]
        train_mean_nodes, _ = model.encode(train_dgl_views, train_features, train_batch_size)
        train_graph_vectors = []
        for v_graph in train_dgl_views:
            v_graph.ndata['z_mean'] = train_mean_nodes
            train_graph_vectors.append(dgl.mean_nodes(v_graph, 'z_mean'))
            del v_graph.ndata['z_mean']
        train_embeddings = torch.cat(train_graph_vectors, dim=1)
        
        # ### CHANGED ### - OPTIMIZATION 1: Prepare a list of individual graphs ONCE
        # Instead of a single batched graph, we create a list of graphs. This avoids
        # the costly unbatching inside the training loop.
        train_dgl_graph_list = dgl.unbatch(train_dgl_views[0])

        # --- Get Test Data ---
        test_dgl_views = []
        for v in range(args.num_views):
            scipy_graphs = [sp.csr_matrix(g[v].cpu().numpy()) for g in list_test_graphs.adj_s]
            dgl_graphs = [dgl.from_scipy(sg) for sg in scipy_graphs]
            test_dgl_views.append(dgl.batch(dgl_graphs).to(device))
        test_features = torch.stack(list_test_graphs.x_s).reshape(-1, list_test_graphs.feature_size).to(device)
        test_batch_size = [len(list_test_graphs.adj_s), list_test_graphs.max_num_nodes]
        test_mean_nodes, _ = model.encode(test_dgl_views, test_features, test_batch_size)
        test_graph_vectors = []
        for v_graph in test_dgl_views:
            v_graph.ndata['z_mean'] = test_mean_nodes
            test_graph_vectors.append(dgl.mean_nodes(v_graph, 'z_mean'))
            del v_graph.ndata['z_mean']
        test_embeddings = torch.cat(test_graph_vectors, dim=1)
        test_dgl_graph_batched = test_dgl_views[0]

    print(f"Generated {train_embeddings.shape[0]} training embeddings (dim={train_embeddings.shape[1]}) and {test_embeddings.shape[0]} test embeddings.")
    logging.info(f"Generated {train_embeddings.shape[0]} training embeddings (dim={train_embeddings.shape[1]}) and {test_embeddings.shape[0]} test embeddings.")
    
    # --- SANITY CHECKS with Simple Classifiers ---
    X_train = train_embeddings.cpu().numpy()
    y_train = train_labels.cpu().numpy().ravel()
    X_test = test_embeddings.cpu().numpy()
    y_test = test_labels.cpu().numpy().ravel()

    print("\n--- Running Sanity Checks on VAE Embeddings ---")
    logging.info("\n--- Running Sanity Checks on VAE Embeddings ---")
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score_sklearn(y_test, y_pred_proba_lr)
    print(f"--- Logistic Regression AUC: {lr_auc:.4f} ---")
    logging.info(f"--- Logistic Regression AUC: {lr_auc:.4f} ---")

    # SVM Classifier
    svm_pipeline = make_pipeline(StandardScaler(), SVC(class_weight='balanced', probability=True, random_state=42))
    svm_pipeline.fit(X_train, y_train)
    y_pred_proba_svm = svm_pipeline.predict_proba(X_test)[:, 1]
    svm_auc = roc_auc_score_sklearn(y_test, y_pred_proba_svm)
    print(f"--- SVM Classifier AUC: {svm_auc:.4f} ---")
    logging.info(f"--- SVM Classifier AUC: {svm_auc:.4f} ---")

    # Random Forest
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
    
    # --- NEW: Evaluation Function ---
    # def evaluate_diffusion_model(model, test_graph, test_embeds, test_lbls):
    #     model.eval()
    #     with torch.no_grad():
    #         predicted_logits = model.sample(test_graph, test_embeds)
        
    #     labels_cpu = test_lbls.cpu().numpy().ravel()
    #     logits_cpu = predicted_logits.cpu().numpy()
    #     preds_cpu = (logits_cpu > 0.5).astype(int)

    #     # acc = accuracy_score(labels_cpu, preds_cpu)
    #     # f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    #     try:
    #         macro_auc = roc_auc_score(labels_cpu, logits_cpu, average='macro')
    #         micro_auc = roc_auc_score(labels_cpu, logits_cpu, average='micro')
    #     except ValueError:
    #         macro_auc = -1.0 
    #         micro_auc = -1.0

    #     acc = 0
    #     f1 = 0

    #     return acc, f1, macro_auc, micro_auc
    def evaluate_diffusion_model(model, test_graph, test_embeds, test_lbls):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
    batch_size = 32
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
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss, _ = diffusion_classifier(batch_dgl_graph, batch_embeddings, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
        
        # --- Consolidated Logging and Evaluation ---
        # Check if it's time for an evaluation
        if (epoch + 1) % eval_every_epochs == 0 or (epoch + 1) == num_diffusion_epochs:
            # 1. Calculate average training loss for the epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # 2. Get evaluation metrics from the test set
            test_acc, test_f1, test_macro_auc, test_micro_auc = evaluate_diffusion_model(
                model=diffusion_classifier,
                test_graph=test_dgl_graph_batched,
                test_embeds=test_embeddings,
                test_lbls=test_labels
            )
            
            # 3. Print and log everything together in a single, clear line
            log_message = (f"Epoch {epoch+1}/{num_diffusion_epochs} | "
                           f"Loss: {avg_loss:.4f} | "
                           f"Test Acc: {test_acc:.4f} | "
                           f"Test F1: {test_f1:.4f} | "
                           f"Test AUC: {test_macro_auc:.4f} | {test_micro_auc:.4f}")
            print(log_message)
            logging.info(log_message)


    print("\n--- Final Diffusion Model Evaluation ---")
    logging.info("\n--- Final Diffusion Model Evaluation ---")
    # The final evaluation is already performed by the loop on the last epoch
    # We can just print a marker that training is complete.
    print("Training finished.")
    logging.info("Training finished.")



# if args.task == "graphClassification":
#     print("\n--- Starting Stage 2: Multi-Label Classification ---")

#     # 1. Prepare data loaders for train and test sets
#     # The `list_graphs` object holds the training data, `list_test_graphs` holds the test data.
#     train_labels = torch.tensor(np.array(list_graphs.labels))
#     test_labels = torch.tensor(np.array(list_test_graphs.labels))
#     num_classes = train_labels.shape[1] # Number of columns in your labels.csv

#     # 2. Generate embeddings for the entire dataset
#     print("Generating embeddings for train and test sets...")
#     model.eval()
#     with torch.no_grad():
#         # Get Train Embeddings
#         # (This is a simplified loop; for large datasets, use a DataLoader)
#         train_dgl_views = []
#         for v in range(args.num_views):
#             train_dgl_views.append(dgl.batch([dgl.from_scipy(sp.csr_matrix(g[v].numpy())) for g in list_graphs.adj_s]).to(device))
#         train_features = torch.cat(list_graphs.x_s).reshape(-1, list_graphs.feature_size).to(device)
#         train_batch_size = [len(list_graphs.adj_s), list_graphs.max_num_nodes]
#         train_embeddings, _ = model.encode(train_dgl_views, train_features, train_batch_size)

#         # Get Test Embeddings
#         test_dgl_views = []
#         for v in range(args.num_views):
#             test_dgl_views.append(dgl.batch([dgl.from_scipy(sp.csr_matrix(g[v].numpy())) for g in list_test_graphs.adj_s]).to(device))
#         test_features = torch.cat(list_test_graphs.x_s).reshape(-1, list_test_graphs.feature_size).to(device)
#         test_batch_size = [len(list_test_graphs.adj_s), list_test_graphs.max_num_nodes]
#         test_embeddings, _ = model.encode(test_dgl_views, test_features, test_batch_size)
    
#     # 3. Define and train the classifier
#     classifier = nn.Sequential(
#         nn.Linear(graphEmDim, 512),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(512, num_classes)
#     ).to(device)

#     # For multi-label classification, use BCEWithLogitsLoss
#     classification_loss_fn = nn.BCEWithLogitsLoss()
#     classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

#     print("Training the classifier...")
#     # Simplified training loop for the classifier
#     for cls_epoch in range(200): # Number of epochs to train the classifier
#         classifier.train()
#         classifier_optimizer.zero_grad()
        
#         # Forward pass
#         predictions = classifier(train_embeddings.to(device))
#         loss = classification_loss_fn(predictions, train_labels.to(device))
        
#         loss.backward()
#         classifier_optimizer.step()
        
#         if (cls_epoch + 1) % 20 == 0:
#             print(f"Classifier Epoch {cls_epoch+1}, Loss: {loss.item():.4f}")

#     # 4. Evaluate the classifier
#     print("Evaluating the classifier on the test set...")
#     classifier.eval()
#     with torch.no_grad():
#         test_predictions_logits = classifier(test_embeddings.to(device))
#         # Apply sigmoid to get probabilities and threshold at 0.5 for predictions
#         test_predictions = (torch.sigmoid(test_predictions_logits) > 0.5).float()

#     # Calculate metrics
#     from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#     test_labels_cpu = test_labels.cpu().numpy()
#     test_predictions_cpu = test_predictions.cpu().numpy()

#     # Accuracy (Exact Match Ratio)
#     accuracy = accuracy_score(test_labels_cpu, test_predictions_cpu)
#     print(f"Exact Match Ratio (Accuracy): {accuracy:.4f}")

#     # F1 Score (Micro and Macro)
#     f1_micro = f1_score(test_labels_cpu, test_predictions_cpu, average='micro')
#     f1_macro = f1_score(test_labels_cpu, test_predictions_cpu, average='macro')
#     print(f"F1 Score (Micro): {f1_micro:.4f}")
#     print(f"F1 Score (Macro): {f1_macro:.4f}")

#     # AUC Score (if labels are not too sparse)
#     try:
#         auc_micro = roc_auc_score(test_labels_cpu, test_predictions_logits.cpu().numpy(), average='micro')
#         auc_macro = roc_auc_score(test_labels_cpu, test_predictions_logits.cpu().numpy(), average='macro')
#         print(f"AUC Score (Micro): {auc_micro:.4f}")
#         print(f"AUC Score (Macro): {auc_macro:.4f}")
#     except ValueError as e:
#         print(f"Could not compute AUC score: {e}")#new

# # graph Classification
# if task == "graphClasssification":
#
#
#     org_adj,x_s, node_num, subgraphs_indexes,  labels = list_graphs.adj_s, list_graphs.x_s, list_graphs.num_nodes, list_graphs.subgraph_indexes, list_graphs.labels
#
#     if(type(decoder))in [  GraphTransformerDecoder_FC]: #
#         node_num = len(node_num)*[list_graphs.max_num_nodes]
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     mean, std = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#
#     prior_samples = model.reparameterize(mean, std)
#     # model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#     # _, prior_samples, _, _, _,_ = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)
#
#
#
#     import classification as CL
#
#     # NN Classifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.NN(prior_samples.cpu().detach(), labels)
#
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
#
#     # KNN clasiifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.knn(prior_samples.cpu().detach(), labels)
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
# # evaluatin graph statistics in graph generation tasks
#
#
# if task == "GraphRepresentation":
#
#     list_test_graphs.processALL(self_for_none=self_for_none)
#
#     test_adj_list = list_test_graphs.get_adj_list()
#     graphFeatures, _ = get_subGraph_features(test_adj_list, None, kernel_model)
#     list_test_graphs.set_features(graphFeatures)
#
#     from_ = 0
#     ro = [-1]
#     org_adj = list_test_graphs.adj_s[from_:to_]
#     x_s = list_test_graphs.x_s[from_:to_]
#     # test_adj_list.num_nodes[from_:to_]
#     labels = list_test_graphs.labels
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#     # else:
#     #     target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)
#
#     # target_kelrnel_val = kernel_model(org_adj, node_num)
#
#     # batchSize = [org_adj.shape[0], org_adj.shape[1]]
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
#
#     reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
#         org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
#
#     i = 0
#     dic = {}
#     digit_labels = []
#     for labl in labels:
#         if labl not in dic:
#             dic[labl] = i
#             i += 1
#         digit_labels.append(dic[labl])
#
#     plotter.featureVisualizer(prior_samples.detach().cpu().numpy(), digit_labels)
