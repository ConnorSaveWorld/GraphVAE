import Aggregation
import dgl
from util import *
import torch
import torch.nn as nn
import numpy as np
import Aggregation
import dgl
from util import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, LayerNorm

class DynamicCouplingLayer(nn.Module):
    """
    A layer to dynamically couple and fuse representations from two modalities (views).
    It generates gates based on the combined information from both views to control
    the information flow, and then fuses the gated representations to update
    the original view-specific representations.
    """
    def __init__(self, hidden_dim, dim_coupling, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_coupling = dim_coupling  # Dimension of the coupling representation C

        # MLP to generate coupling representation C_i from [H_view1 || H_view2]
        self.mlp_couple = Sequential(
            Linear(hidden_dim * 2, dim_coupling),
            nn.GELU(),
            Dropout(dropout_rate)
        )

        # MLPs to generate gates from the coupling representation C
        self.gate_mlp_v1 = Linear(dim_coupling, hidden_dim)
        self.gate_mlp_v2 = Linear(dim_coupling, hidden_dim)

        # MLP for fusing the gated representations
        self.fusion_mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # Input is [gated_H_v1 || gated_H_v2]
            nn.GELU(),
            Dropout(dropout_rate)
        )

        # MLPs to project the fused information back to update each view's representation
        self.v1_update_mlp = Linear(hidden_dim, hidden_dim)
        self.v2_update_mlp = Linear(hidden_dim, hidden_dim)

        self.norm_v1 = LayerNorm(hidden_dim)
        self.norm_v2 = LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = Dropout(dropout_rate)

    def forward(self, h_v1, h_v2):
        # h_v1, h_v2: (total_nodes_in_batch, hidden_dim)

        # 1. Generate Coupling Representation C
        combined_hidden = torch.cat((h_v1, h_v2), dim=-1)
        C = self.mlp_couple(combined_hidden)

        # 2. Generate and Apply Gates
        gate_v1 = torch.sigmoid(self.gate_mlp_v1(C))
        gate_v2 = torch.sigmoid(self.gate_mlp_v2(C))
        h_v1_gated = gate_v1 * h_v1
        h_v2_gated = gate_v2 * h_v2

        # 3. Fuse Gated Representations
        h_fused_input = torch.cat((h_v1_gated, h_v2_gated), dim=-1)
        h_fused = self.fusion_mlp(h_fused_input)

        # 4. Update Modality Representations with Residual Connections
        h_v1_update_projection = self.dropout(self.act(self.v1_update_mlp(h_fused)))
        h_v1_new = self.norm_v1(h_v1 + h_v1_update_projection)

        h_v2_update_projection = self.dropout(self.act(self.v2_update_mlp(h_fused)))
        h_v2_new = self.norm_v2(h_v2 + h_v2_update_projection)

        return h_v1_new, h_v2_new


class ChannelAttentionGraph(nn.Module):
    """CBAM-inspired channel attention for graph node features"""
    def __init__(self, hidden_dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (N, D) where N is number of nodes
        b, c = x.size()
        
        # Global average and max pooling across all nodes
        avg_out = self.fc(x.mean(dim=0, keepdim=True))  # (1, D)
        max_out = self.fc(x.max(dim=0, keepdim=True)[0])  # (1, D)
        
        # Channel attention weights
        attention = self.sigmoid(avg_out + max_out)  # (1, D)
        
        return x * attention.expand_as(x)


# >>> ADD START: GraphSpatialAttention and GraphCBAM for graph data <<<
class GraphSpatialAttention(nn.Module):
    """Spatial attention over graph nodes (treats nodes as the spatial dimension)."""
    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        # Use 1D convolution where the sequence length is the number of nodes
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ln = nn.LayerNorm(hidden_dim)  # stabilise stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_nodes_in_batch, D)
        x_norm = self.ln(x)  # normalize before spatial attention
        avg_out = x_norm.mean(dim=1, keepdim=True)  # (N,1)
        max_out, _ = x_norm.max(dim=1, keepdim=True)  # (N,1)
        y = torch.cat([avg_out, max_out], dim=1)  # (N,2)
        # Conv1d expects (B, C, L). Treat batch=1, C=2, L=N
        y = y.t().unsqueeze(0)  # (1,2,N)
        y = self.sigmoid(self.conv(y))  # (1,1,N)
        y = y.squeeze(0).t()  # (N,1)
        return x * y.expand_as(x)


class GraphCBAM(nn.Module):
    """Convolutional Block Attention Module adapted for graph node features with residual gating."""
    def __init__(self, hidden_dim: int, reduction: int = 8, spatial_kernel: int = 3):
        super().__init__()
        self.channel_att = ChannelAttentionGraph(hidden_dim, reduction)
        self.spatial_att = GraphSpatialAttention(hidden_dim, kernel_size=spatial_kernel)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable gate initialized to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention first (as before)
        x = self.channel_att(x)
        # Add spatial attention as a residual refinement with learnable gating
        return x + self.gamma * self.spatial_att(x)
# >>> ADD END <<<


class GraphNonLocalBlock(nn.Module):
    """Non-local block adapted for graph node features"""
    def __init__(self, hidden_dim, reduction=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inter_dim = hidden_dim // reduction
        
        self.theta = nn.Linear(hidden_dim, self.inter_dim, bias=False)
        self.phi = nn.Linear(hidden_dim, self.inter_dim, bias=False)
        self.g = nn.Linear(hidden_dim, self.inter_dim, bias=False)
        self.out_proj = nn.Linear(self.inter_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x: (N, D) where N is number of nodes
        batch_size, _ = x.size()
        
        # Generate query, key, value
        q = self.theta(x)  # (N, D//2)
        k = self.phi(x)    # (N, D//2)
        v = self.g(x)      # (N, D//2)
        
        # Attention matrix
        attention = torch.matmul(q, k.transpose(-2, -1))  # (N, N)
        attention = self.softmax(attention / (self.inter_dim ** 0.5))
        
        # Apply attention
        out = torch.matmul(attention, v)  # (N, D//2)
        out = self.out_proj(out)  # (N, D)
        
        # Residual connection
        return x + out


# In model.py

class DynamicCouplingEncoder(nn.Module):
    def __init__(self, in_feature_dim, num_views, hidden_dim, num_initial_layers,
                 num_coupling_layers, dim_coupling, dropout_rate, GraphLatentDim, nhead=4): # Add nhead
        super(DynamicCouplingEncoder, self).__init__()
        
        assert num_views == 2, "DynamicCouplingEncoder currently supports only 2 views."
        
        self.num_views = num_views
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.initial_embedding = Linear(in_feature_dim, hidden_dim)

        # --- UPDATED: Intra-Modal GNNs using GATv2Conv ---
        self.initial_gnns_v1 = nn.ModuleList()
        self.initial_gnns_v2 = nn.ModuleList()
        for _ in range(num_initial_layers):
            self.initial_gnns_v1.append(dgl.nn.pytorch.conv.GATv2Conv(
                hidden_dim, hidden_dim // nhead, nhead, feat_drop=dropout_rate, attn_drop=dropout_rate, activation=F.relu
            ))
            self.initial_gnns_v2.append(dgl.nn.pytorch.conv.GATv2Conv(
                hidden_dim, hidden_dim // nhead, nhead, feat_drop=dropout_rate, attn_drop=dropout_rate, activation=F.relu
            ))
        
        self.initial_norms_v1 = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_initial_layers)])
        self.initial_norms_v2 = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_initial_layers)])

        # --- Inter-Modal Dynamically Coupled Layers (no change here) ---
        self.coupling_layers = nn.ModuleList()
        for _ in range(num_coupling_layers):
            self.coupling_layers.append(
                DynamicCouplingLayer(hidden_dim, dim_coupling, dropout_rate)
            )

        # --- Enhanced Attention Mechanisms ---
        # Use CBAM (channel + spatial) followed by non-local block
        self.graph_cbam = GraphCBAM(hidden_dim * 2, reduction=8, spatial_kernel=3)
        self.nonlocal_block = GraphNonLocalBlock(hidden_dim * 2, reduction=2)

        # --- Final Fusion and Projection for VAE (no change here) ---
        final_fusion_dim = hidden_dim * 2
        self.final_projection = Linear(final_fusion_dim, GraphLatentDim)
        
        self.stochastic_mean_layer = node_mlp(GraphLatentDim, [GraphLatentDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatentDim, [GraphLatentDim])
        
        # --- Multi-Strategy Graph Pooling ---
        # Project concatenated mean/max/sum pooling (3 * GraphLatentDim) back to GraphLatentDim
        self.pooling_projection = nn.Sequential(
            nn.Linear(GraphLatentDim * 3, GraphLatentDim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.activation = nn.GELU()

    def forward(self, graph_list, features, batchSize):
        h = self.initial_embedding(features)
        h_v1, h_v2 = h, h

        for i in range(len(self.initial_gnns_v1)):
            h_v1_res = h_v1
            # .flatten(1) is needed after GAT to combine head outputs
            h_v1 = self.initial_gnns_v1[i](graph_list[0], h_v1).flatten(1)
            h_v1 = self.initial_norms_v1[i](h_v1 + h_v1_res)

            h_v2_res = h_v2
            h_v2 = self.initial_gnns_v2[i](graph_list[1], h_v2).flatten(1)
            h_v2 = self.initial_norms_v2[i](h_v2 + h_v2_res)

        for layer in self.coupling_layers:
            h_v1, h_v2 = layer(h_v1, h_v2)

        # Apply enhanced attention after coupling
        h_combined = torch.cat((h_v1, h_v2), dim=-1)
        h_combined = self.graph_cbam(h_combined)
        h_combined = self.nonlocal_block(h_combined)
        h_v1, h_v2 = h_combined[:, :self.hidden_dim], h_combined[:, self.hidden_dim:]

        h_final_fused = torch.cat((h_v1, h_v2), dim=-1)
        h_final = self.activation(self.final_projection(h_final_fused))


        #(NEW new New)
        final_node_embeddings = self.activation(self.final_projection(h_final_fused))
        # --- GRAPH-LEVEL POOLING (Learnable Attention) ---
        batched_graph = graph_list[0]
        
        # Multi-strategy pooling: mean + max + sum for robustness
        batched_graph.ndata['h_temp'] = final_node_embeddings
        mean_pool = dgl.mean_nodes(batched_graph, 'h_temp')
        max_pool = dgl.max_nodes(batched_graph, 'h_temp')
        sum_pool = dgl.sum_nodes(batched_graph, 'h_temp')
        del batched_graph.ndata['h_temp']
        
        # Combine pooling strategies
        combined_pool = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        graph_level_embedding = self.pooling_projection(combined_pool)
        
        # --- Generate distribution from the GRAPH-LEVEL embedding ---
        mean = self.stochastic_mean_layer(graph_level_embedding, activation=lambda x: x)
        log_std = self.stochastic_log_std_layer(graph_level_embedding, activation=lambda x: x)
        # The output of the forward pass for this encoder is now a tuple
        # This will be used by StagedSupervisedVAE
        return final_node_embeddings, mean, log_std
        
        # mean = self.stochastic_mean_layer(h_final, activation=lambda x: x)
        # log_std = self.stochastic_log_std_layer(h_final, activation=lambda x: x)

        # return mean, log_std


class AveEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers=[256, 256, 256], GraphLatntDim=1024):
        super(AveEncoder, self).__init__()

        hiddenLayers = [in_feature_dim] + hiddenLayers + [GraphLatntDim]
        self.normLayers = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hiddenLayers[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
        self.normLayers.append(torch.nn.LayerNorm(hiddenLayers[-1], elementwise_affine=False))
        self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1],
                                                                            activation=None, bias=True, weight=True) for
                                              i in range(len(hiddenLayers) - 1)])

        self.poolingLayer = Aggregation.AvePool()

        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])

    def forward(self, graph, features, batchSize, activation=torch.nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(self.GCNlayers)):
            h = self.GCNlayers[i](graph, h)
            h = activation(h)
            # if((i<len(self.GCNlayers)-1)):
            h = self.normLayers[i](h)

        h = h.reshape(*batchSize, -1)

        h = self.poolingLayer(h)

        h = self.normLayers[-1](h)
        mean = self.stochastic_mean_layer(h, activation=lambda x: x)
        log_std = self.stochastic_log_std_layer(h, activation=lambda x: x)

        return mean, log_std


# In model.py

class GraphTransformerDecoder_FC(torch.nn.Module):
    def __init__(self, input_dim, lambdaDim, SubGraphNodeNum, num_views, directed=True):
        """
        The updated __init__ function for the FC decoder.
        """
        super(GraphTransformerDecoder_FC, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        self.num_views = num_views
        self.directed = directed

        # --- KEY CHANGE IS HERE ---
        # The true input dimension to the decoder is now the graph embedding dimension
        # multiplied by the number of views, because we are concatenating.
        concatenated_input_dim = input_dim * num_views
        
        # Define the layer sizes, starting with the new concatenated dimension.
        layers = [concatenated_input_dim] + [1024, 1024, 1024]

        # The final layer size calculation remains the same.
        if directed:
            final_layer_size = SubGraphNodeNum * SubGraphNodeNum * self.num_views
        else:
            triu_size = int((SubGraphNodeNum * (SubGraphNodeNum - 1) / 2) + SubGraphNodeNum)
            final_layer_size = triu_size * self.num_views
        layers.append(final_layer_size)

        self.normLayers = torch.nn.ModuleList(
            [torch.nn.LayerNorm(layers[i + 1], elementwise_affine=False) for i in range(len(layers) - 2)])
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i + 1], dtype=torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.LeakyReLU(0.001)):
        # The forward pass does not need to be changed.
        for i in range(len(self.layers)):
            in_tensor = self.layers[i](in_tensor)
            if i != len((self.layers)) - 1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)

        if self.directed:
            ADJ_views = in_tensor.reshape(in_tensor.shape[0], self.num_views, self.SubGraphNodeNum, self.SubGraphNodeNum)
        else:
            ADJ_views = torch.zeros((in_tensor.shape[0], self.num_views, self.SubGraphNodeNum, self.SubGraphNodeNum)).to(in_tensor.device)
            flat_views = in_tensor.view(in_tensor.shape[0], self.num_views, -1)
            
            iu_indices = torch.triu_indices(self.SubGraphNodeNum, self.SubGraphNodeNum, offset=0)
            ADJ_views[:, :, iu_indices[0], iu_indices[1]] = flat_views
            ADJ_views = ADJ_views + ADJ_views.transpose(-1, -2)
            diag_indices = torch.arange(self.SubGraphNodeNum)
            ADJ_views[:, :, diag_indices, diag_indices] /= 2

        return ADJ_views


# In model.py

class kernelGVAE(torch.nn.Module):
    def __init__(self, ker, encoder, decoder, AutoEncoder, graphEmDim=4096, num_views=1):
        super(kernelGVAE, self).__init__()
        # Note: 'graphEmDim' now refers to the dimension of EACH NODE's latent vector.
        self.embeding_dim = graphEmDim
        self.kernel = ker
        self.AutoEncoder = AutoEncoder
        self.decode = decoder # This will be an instance of NodeLevelInnerProductDecoder
        self.encode = encoder
        self.num_views = num_views

    def reparameterize(self, mean, log_std):
        if self.AutoEncoder:
            return mean
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, graph_list, features, batchSize, subgraphs_indexes):
        """
        The forward pass for a TRUE Node-Level VAE.
        """
        # 1. ENCODING
        # The encoder returns node embeddings, and graph-level mean and log_std
        node_embeddings, mean, log_std = self.encode(graph_list, features, batchSize)
        
        # 2. SAMPLING
        # We still sample at the graph level for other purposes (like KL divergence)
        # but we will use the direct node_embeddings for reconstruction.
        z_graph = self.reparameterize(mean, log_std)
        
        # 3. DECODING
        # The decoder takes the batched graph structure and the NODE embeddings.
        batched_graph = graph_list[0]
        reconstructed_adj_logit_views = self.decode(batched_graph, node_embeddings)
        
        reconstructed_adj_views = torch.sigmoid(reconstructed_adj_logit_views)

        # 4. KERNEL CALCULATION (no change)
        kernel_value_views = []
        if self.kernel is not None and self.kernel.kernel_type:
            # This part needs to handle the fact that reconstructed views are padded
            # to the max size *in the batch*, not the max size of the dataset.
            max_nodes_in_batch = reconstructed_adj_views.shape[-1]
            for v in range(self.num_views):
                reconstructed_single_view = reconstructed_adj_views[:, v, :max_nodes_in_batch, :max_nodes_in_batch]
                kernel_value_views.append(self.kernel(reconstructed_single_view))
        
        # Return the PER-NODE distributions and a placeholder for the aggregated samples
        # The second return value (prior_samples) is not well-defined here, so we can return None or z_nodes
        return reconstructed_adj_views, None, mean, log_std, kernel_value_views, reconstructed_adj_logit_views


### MODIFICATION 3: Use concatenation for multi-view fusion
class MultiViewAveEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, num_views, hiddenLayers=[256, 256], GraphLatntDim=1024):
        super(MultiViewAveEncoder, self).__init__()
        self.num_views = num_views

        # --- View-Specific Layers (Early Fusion) ---
        self.view_specific_gcns = nn.ModuleList()
        for _ in range(self.num_views):
            self.view_specific_gcns.append(
                dgl.nn.pytorch.conv.GraphConv(
                    in_feature_dim,
                    hiddenLayers[0],
                    activation=None,
                    bias=True,
                    weight=True
                )
            )

        # --- Shared Layers ---
        # The input dimension to the first shared layer is now num_views * hiddenLayers[0]
        # because we are concatenating the features from each view.
        concatenated_dim = hiddenLayers[0] * self.num_views
        shared_hidden = [concatenated_dim] + hiddenLayers[1:] + [GraphLatntDim]
        
        self.shared_GCNlayers = nn.ModuleList([
            dgl.nn.pytorch.conv.GraphConv(shared_hidden[i], shared_hidden[i+1],
                                          activation=None, bias=True, weight=True)
            for i in range(len(shared_hidden) - 1)
        ])
        
        # We need norm layers for the concatenated dimension as well.
        self.normLayers = nn.ModuleList([
            nn.LayerNorm(concatenated_dim)
        ] + [
            nn.LayerNorm(dim) for dim in hiddenLayers[1:] + [GraphLatntDim]
        ])
        
        self.poolingLayer = Aggregation.AvePool()
        
        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, graph_list, features, batchSize):
        # 1. Apply view-specific GCNs
        view_outputs = []
        for i in range(self.num_views):
            h_view = self.view_specific_gcns[i](graph_list[i], features)
            view_outputs.append(h_view)
        
        # 2. Fuse the outputs by concatenation along the feature dimension (dim=1)
        h = torch.cat(view_outputs, dim=1)
        h = self.activation(h)
        h = self.normLayers[0](h)

        # 3. Pass through shared GCN layers
        # The graph structure from the first view is used for message passing.
        graph_structure_for_shared_layers = graph_list[0]
        for i in range(len(self.shared_GCNlayers)):
            h = self.shared_GCNlayers[i](graph_structure_for_shared_layers, h)
            h = self.activation(h)
            h = self.normLayers[i+1](h)

        # # 4. Reshape and Pool
        # h = h.reshape(batchSize[0], batchSize[1], -1)
        # h = self.poolingLayer(h)
        # h = self.normLayers[-1](h)

        # 5. Get mean and log_std
        mean = self.stochastic_mean_layer(h, activation=lambda x: x)
        log_std = self.stochastic_log_std_layer(h, activation=lambda x: x)

        return h, mean, log_std

# class AveEncoder(torch.nn.Module):
#     def __init__(self, in_feature_dim, hiddenLayers=[256, 256, 256], GraphLatntDim=1024):
#         super(AveEncoder, self).__init__()

#         hiddenLayers = [in_feature_dim] + hiddenLayers + [GraphLatntDim]
#         self.normLayers = torch.nn.ModuleList(
#             [torch.nn.LayerNorm(hiddenLayers[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
#         self.normLayers.append(torch.nn.LayerNorm(hiddenLayers[-1], elementwise_affine=False))
#         self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1],
#                                                                             activation=None, bias=True, weight=True) for
#                                               i in range(len(hiddenLayers) - 1)])

#         self.poolingLayer = Aggregation.AvePool()

#         self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
#         self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])

#     def forward(self, graph, features, batchSize, activation=torch.nn.LeakyReLU(0.01)):
#         h = features
#         for i in range(len(self.GCNlayers)):
#             h = self.GCNlayers[i](graph, h)
#             h = activation(h)
#             # if((i<len(self.GCNlayers)-1)):
#             h = self.normLayers[i](h)

#         h = h.reshape(*batchSize, -1)

#         h = self.poolingLayer(h)

#         h = self.normLayers[-1](h)
#         mean = self.stochastic_mean_layer(h, activation=lambda x: x)
#         log_std = self.stochastic_log_std_layer(h, activation=lambda x: x)

#         return mean, log_std


# class GraphTransformerDecoder_FC(torch.nn.Module):
#     def __init__(self, input_dim, lambdaDim, SubGraphNodeNum, num_views, directed=True):
#         super(GraphTransformerDecoder_FC, self).__init__()
#         self.SubGraphNodeNum = SubGraphNodeNum
#         self.num_views = num_views # <<< NEW
#         self.directed = directed
#         layers = [input_dim] + [1024, 1024, 1024]

#         if directed:
#             final_layer_size = SubGraphNodeNum * SubGraphNodeNum * self.num_views
#         else:
#             triu_size = int((SubGraphNodeNum * (SubGraphNodeNum - 1) / 2) + SubGraphNodeNum)
#             final_layer_size = triu_size * self.num_views
#         layers.append(final_layer_size)

#         self.normLayers = torch.nn.ModuleList(
#             [torch.nn.LayerNorm(layers[i + 1], elementwise_affine=False) for i in range(len(layers) - 2)])
#         self.layers = torch.nn.ModuleList(
#             [torch.nn.Linear(layers[i], layers[i + 1], torch.float32) for i in range(len(layers) - 1)])

#     def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.LeakyReLU(0.001)):

#         for i in range(len(self.layers)):
#             # in_tensor = self.normLayers[i](in_tensor)
#             in_tensor = self.layers[i](in_tensor)
#             if i != len((self.layers)) - 1:
#                 in_tensor = activation(in_tensor)
#                 in_tensor = self.normLayers[i](in_tensor)

#         # Reshape to get K adjacency matrices
#         if self.directed:
#             # Shape: (batch_size, num_views, N, N)
#             ADJ_views = in_tensor.reshape(in_tensor.shape[0], self.num_views, self.SubGraphNodeNum, self.SubGraphNodeNum)
#         else:
#             # Handle the undirected case for multiple views
#             ADJ_views = torch.zeros((in_tensor.shape[0], self.num_views, self.SubGraphNodeNum, self.SubGraphNodeNum)).to(in_tensor.device)
#         flat_views = in_tensor.view(in_tensor.shape[0], self.num_views, -1)

#         # Get indices for the upper triangle (excluding the diagonal)
#         triu_indices = torch.triu_indices(self.SubGraphNodeNum, self.SubGraphNodeNum, offset=1)
        
#         # Get diagonal indices
#         diag_indices = torch.arange(self.SubGraphNodeNum)

#         for v in range(self.num_views):
#             # Fill the upper triangle
#             upper_tri_size = triu_indices.shape[1]
#             ADJ_views[:, v, triu_indices[0], triu_indices[1]] = flat_views[:, v, :upper_tri_size]
            
#             # Fill the diagonal
#             ADJ_views[:, v, diag_indices, diag_indices] = flat_views[:, v, upper_tri_size:]

#         # Symmetrize by adding the transpose. Diagonal is now correct.
#         ADJ_views = ADJ_views + ADJ_views.transpose(-1, -2)

#         return ADJ_views


#         # if self.directed:
#         #     ADJ = in_tensor.reshape(in_tensor.shape[0], self.SubGraphNodeNum, -1)
#         # else:
#         #     ADJ = torch.zeros((in_tensor.shape[0], self.SubGraphNodeNum, self.SubGraphNodeNum)).to(in_tensor.device)
#         #     ADJ[:, torch.tril_indices(self.SubGraphNodeNum, self.SubGraphNodeDim, -1)[0],
#         #     torch.tril_indices(self.SubGraphNodeNum, self.SubGraphNodeDim, -1)[1]] = in_tensor[:,
#         #                                                                    :(in_tensor.shape[-1]) - self.SubGraphNodeNum]
#         #     ADJ = ADJ + ADJ.permute(0, 2, 1)
#         #     ind = np.diag_indices(ADJ.shape[-1])
#         #     ADJ[:, ind[0], ind[1]] = in_tensor[:, -self.SubGraphNodeNum:]  # torch.ones(ADJ.shape[-1]).to(ADJ.device)
#         # # adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
#         # # return adj_list
#         # # if subgraphs_indexes==None:
#         # adj_list= torch.matmul(in_tensor,in_tensor.permute(0,2,1))
#         # return ADJ
#         # # else:
#         # #     adj_list = []
#         # #     for i in range(in_tensor.shape[0]):
#         # #         adj_list.append(torch.matmul(in_tensor[i][subgraphs_indexes[i]].to(device), in_tensor[i][subgraphs_indexes[i]].permute(0,2,1)).to(device))
#         # #     return torch.stack(adj_list)


# class kernelGVAE(torch.nn.Module):
#     def __init__(self, ker, encoder, decoder, AutoEncoder, graphEmDim=4096, num_views=1):
#         super(kernelGVAE, self).__init__()
#         self.embeding_dim = graphEmDim
#         self.kernel = ker  # TODO: bin and width whould be determined if kernel is his
#         self.AutoEncoder = AutoEncoder
#         self.decode = decoder
#         self.encode = encoder
#         self.num_views = num_views # <<< NEW

#         self.stochastic_mean_layer = node_mlp(self.embeding_dim, [self.embeding_dim])
#         self.stochastic_log_std_layer = node_mlp(self.embeding_dim, [self.embeding_dim])

#     def forward(self, graph_list, features, batchSize, subgraphs_indexes):
#         """
#         :param graph_list: A list of batched DGL graphs, one for each view.
#         :param features: Normalized node feature matrix.
#         """
#         # The encoder now takes a list of graphs
#         mean, log_std = self.encode(graph_list, features, batchSize)
        
#         samples = self.reparameterize(mean, log_std)
        
#         # The decoder returns logits for all views
#         reconstructed_adj_logit_views = self.decode(samples, subgraphs_indexes)
#         reconstructed_adj_views = torch.sigmoid(reconstructed_adj_logit_views)

#         # Kernel calculation needs to be applied to each reconstructed view
#         kernel_value_views = []
#         if self.kernel is not None and self.kernel.kernel_type:
#             for v in range(self.num_views):
#                 # self.kernel expects a batch of single-view graphs
#                 reconstructed_single_view = reconstructed_adj_views[:, v, :, :]
#                 kernel_value_views.append(self.kernel(reconstructed_single_view))
        
#         return reconstructed_adj_views, samples, mean, log_std, kernel_value_views, reconstructed_adj_logit_views

#     def reparameterize(self, mean, log_std):
#         if self.AutoEncoder:
#             return mean
#         # # Your existing reparameterization logic is correct
#         # var = torch.exp(log_std).pow(2)
#         # eps = torch.randn_like(var)
#         # sample = eps.mul(var.sqrt()).add_(mean) # A small correction: multiply by std (sqrt(var)), not var
#         # return sample
#         std = torch.exp(0.5 * log_std)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mean)


# class MultiViewAveEncoder(torch.nn.Module):
#     def __init__(self, in_feature_dim, num_views, hiddenLayers=[256, 256], GraphLatntDim=1024):
#         super(MultiViewAveEncoder, self).__init__()
#         self.num_views = num_views

#         # --- View-Specific Layers (Early Fusion) ---
#         # Create a separate GCN layer for each view (e.g., SC and FC)
#         self.view_specific_gcns = nn.ModuleList()
#         for _ in range(self.num_views):
#             self.view_specific_gcns.append(
#                 dgl.nn.pytorch.conv.GraphConv(
#                     in_feature_dim,
#                     hiddenLayers[0],
#                     activation=None,
#                     bias=True,
#                     weight=True
#                 )
#             )

#         # --- Shared Layers ---
#         # The rest of the architecture is shared
#         shared_hidden = [hiddenLayers[0]] + hiddenLayers[1:] + [GraphLatntDim]
        
#         self.shared_GCNlayers = nn.ModuleList([
#             dgl.nn.pytorch.conv.GraphConv(shared_hidden[i], shared_hidden[i+1],
#                                           activation=None, bias=True, weight=True)
#             for i in range(len(shared_hidden) - 1)
#         ])
        
#         self.normLayers = nn.ModuleList([
#             nn.LayerNorm(dim) for dim in hiddenLayers + [GraphLatntDim]
#         ])
        
#         self.poolingLayer = Aggregation.AvePool()
        
#         self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
#         self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
#         self.activation = nn.LeakyReLU(0.01)

#     def forward(self, graph_list, features, batchSize):
#         # graph_list is a list of batched DGL graphs, one for each view
        
#         # 1. Apply view-specific GCNs
#         view_outputs = []
#         for i in range(self.num_views):
#             # Apply the i-th GCN to the i-th view graph
#             h_view = self.view_specific_gcns[i](graph_list[i], features)
#             view_outputs.append(h_view)
        
#         # 2. Fuse the outputs by averaging
#         h = torch.stack(view_outputs, dim=0).mean(dim=0)
#         h = self.activation(h)
#         h = self.normLayers[0](h)

#         # 3. Pass through shared GCN layers
#         # We can use the first view's graph structure for subsequent convolutions
#         graph_structure_for_shared_layers = graph_list[0]
#         for i in range(len(self.shared_GCNlayers)):
#             h = self.shared_GCNlayers[i](graph_structure_for_shared_layers, h)
#             h = self.activation(h)
#             h = self.normLayers[i+1](h)

#         # 4. Reshape and Pool
#         h = h.reshape(batchSize[0], batchSize[1], -1)

#         h = self.poolingLayer(h)
#         h = self.normLayers[-1](h)

#         # 5. Get mean and log_std
#         mean = self.stochastic_mean_layer(h, activation=lambda x: x)
#         log_std = self.stochastic_log_std_layer(h, activation=lambda x: x)

#         return mean, log_std


# In model.py or equivalent

# In model.py

class NodeLevelInnerProductDecoder(nn.Module):
    """
    Decodes a batch of graphs from node-level latent embeddings.
    This decoder is designed to work with DGL's batching system.
    """
    def __init__(self, in_node_dim, num_views):
        super(NodeLevelInnerProductDecoder, self).__init__()
        self.num_views = num_views
        
        # Optional: A linear layer to project the latent node embeddings to a new space
        # before taking the inner product. This can add expressive power.
        # Let's keep it simple for now and do a direct inner product.
        # self.projection = nn.Linear(in_node_dim, some_other_dim)

    def forward(self, batched_graph, z_nodes):
        """
        Args:
            batched_graph (DGLGraph): The batched graph structure from DGL.
                                      Needed to un-batch the results.
            z_nodes (Tensor): A tensor of latent embeddings for all nodes in the batch,
                              shape (total_nodes_in_batch, in_node_dim).
        """
        
        # Use DGL to get a list of un-batched graphs.
        # Each element of the list will be one of the original graphs.
        graph_list = dgl.unbatch(batched_graph)
        
        # Get the node counts for each graph in the batch. This is crucial.
        nodes_per_graph = [g.num_nodes() for g in graph_list]
        
        # Use torch.split to divide the single z_nodes tensor back into
        # a list of tensors, one for each graph.
        # z_nodes_per_graph is now a list of tensors, e.g.,
        # [ (N1, D), (N2, D), ... ] where N1, N2 are node counts.
        z_nodes_per_graph = torch.split(z_nodes, nodes_per_graph, dim=0)
        
        adj_logits_list = []
        for node_embeddings in z_nodes_per_graph:
            # For each graph's set of node embeddings:
            # node_embeddings has shape (num_nodes_in_this_graph, in_node_dim)
            
            # Reconstruct adjacency matrix via inner product.
            # (N, D) @ (D, N) -> (N, N)
            adj_logits = torch.matmul(node_embeddings, node_embeddings.transpose(-1, -2))
            adj_logits_list.append(adj_logits)
            
        # The reconstructed logits have variable sizes (N1xN1, N2xN2, ...).
        # We need to pad them to the maximum size to stack them into a single batch tensor.
        max_nodes = batched_graph.batch_size * 0 + max(nodes_per_graph) # Get max nodes in this specific batch
        
        padded_adj_logits_list = []
        for adj_logit in adj_logits_list:
            num_nodes = adj_logit.shape[0]
            # Create a padded tensor of zeros
            padded_logit = torch.zeros(max_nodes, max_nodes, device=z_nodes.device)
            # Copy the actual logit data into the top-left corner
            padded_logit[:num_nodes, :num_nodes] = adj_logit
            padded_adj_logits_list.append(padded_logit)

        # Stack the padded logits into a single tensor for the batch.
        # Shape: (batch_size, max_nodes, max_nodes)
        reconstructed_adj_logits = torch.stack(padded_adj_logits_list, dim=0)
        
        # Finally, expand to multiple views, assuming shared reconstruction.
        # Shape: (batch_size, num_views, max_nodes, max_nodes)
        adj_logits_views = reconstructed_adj_logits.unsqueeze(1).repeat(1, self.num_views, 1, 1)

        return adj_logits_views

class ClassificationDecoder(nn.Module):
    """
    Enhanced MLP decoder with residual connections, attention, and better regularization
    inspired by SOTA VAE architectures.
    """
    def __init__(self, latent_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Enhanced architecture with residual blocks
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Residual blocks for better gradient flow
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # GELU activation for better gradients
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Channel attention for feature refinement
        self.channel_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final classification layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, z_graph):
        # Apply input projection
        h = self.input_proj(z_graph)
        
        # First residual block with layer norm
        h = self.norm1(h)
        residual = h
        h = self.res_block1(h)
        h = h + residual  # Residual connection
        
        # Second residual block with layer norm
        h = self.norm2(h)
        residual = h
        h = self.res_block2(h)
        h = h + residual  # Residual connection
        
        # Apply channel attention
        attention = self.channel_attn(h)
        h = attention * h
        
        # Final classification
        return self.mlp(h)

class StagedSupervisedVAE(nn.Module):
    """
    A VAE designed for two-stage training.
    In Stage 1 (training this model), it performs supervised graph classification.
    Its `encode` method can then be used in Stage 2 as a feature extractor
    for either node or graph embeddings.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, graph_list, features, batchSize):
        """
        This forward pass is for training Stage 1.
        """
        # The encoder should be modified to return both node and graph embeddings
        node_embeddings, graph_mean, graph_log_std = self.encoder(graph_list, features, batchSize)

        # Sample from the graph-level distribution
        z_graph = self.reparameterize(graph_mean, graph_log_std)

        # Decode to get class predictions
        predicted_logits = self.decoder(z_graph)

        return predicted_logits, graph_mean, graph_log_std, node_embeddings