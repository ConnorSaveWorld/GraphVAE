#
# label_diffusion.py (Modified)
#
import torch
import torch.nn as nn
import dgl
from diffusion import DDM, loss_fn, extract

import torch
import torch.nn as nn
import dgl
from diffusion import DDM, loss_fn, extract

class LabelDiffusionClassifier(nn.Module):
    def __init__(self, graph_embedding_dim, DDM_config):
        """
        A wrapper for the DDM model to perform label classification on a full graph view.

        Args:
            graph_embedding_dim (int): The dimension of the conditioning graph embedding.
            DDM_config (dict): A dictionary of configurations for the DDM model.
        """
        super().__init__()
        self.graph_embedding_dim = graph_embedding_dim

        ddm_in_dim = 1
        DDM_config['in_dim'] = ddm_in_dim
        self.ddm = DDM(**DDM_config)

        self.embedding_projection = nn.Linear(graph_embedding_dim, DDM_config["num_hidden"])

    def forward(self, batched_dgl_graph, graph_embeddings, target_labels):
        """ The training forward pass. """
        device = graph_embeddings.device
        
        # Ensure target_labels is (batch_size, 1) before broadcasting
        target_labels_reshaped = target_labels.view(-1, 1)
        labels_as_nodes = dgl.broadcast_nodes(batched_dgl_graph, target_labels_reshaped)

        # DDM Forward Process (Noising)
        t_nodes = torch.randint(self.ddm.T, size=(labels_as_nodes.shape[0],), device=device)
        noise = torch.randn_like(labels_as_nodes)
        noisy_labels_as_nodes = (
            extract(self.ddm.sqrt_alphas_bar, t_nodes, labels_as_nodes.shape) * labels_as_nodes +
            extract(self.ddm.sqrt_one_minus_alphas_bar, t_nodes, labels_as_nodes.shape) * noise
        )

        # The conditioning signal (time + graph embedding) is also broadcast to every node.
        t_per_graph = torch.randint(self.ddm.T, size=(graph_embeddings.shape[0], ), device=device)
        time_embed = self.ddm.time_embedding(t_per_graph)
        embedding_proj = self.embedding_projection(graph_embeddings)
        conditioning_signal_per_graph = time_embed + embedding_proj
        conditioning_signal_nodes = dgl.broadcast_nodes(batched_dgl_graph, conditioning_signal_per_graph)

        # Denoise using the UNet on the actual graph structure.
        predicted_x0_nodes, _ = self.ddm.net(
            batched_dgl_graph, 
            x_t=noisy_labels_as_nodes, 
            time_embed=conditioning_signal_nodes
        )

        # Calculate Loss at the node level.
        loss = loss_fn(predicted_x0_nodes, labels_as_nodes, alpha=self.ddm.alpha_l)
        loss_item = {"loss": loss.item()}

        return loss, loss_item

    @torch.no_grad()
    def sample(self, batched_dgl_graph, graph_embeddings, ddim_steps=50, eta=0.0):
        """ Fast DDIM sampling with much fewer steps. """
        device = graph_embeddings.device
        num_total_nodes = batched_dgl_graph.num_nodes()
        num_graphs = graph_embeddings.shape[0]

        # Create DDIM timestep schedule
        c = self.ddm.T // ddim_steps
        ddim_timesteps = torch.arange(0, self.ddm.T, c, device=device)
        ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([self.ddm.T-1], device=device)])
        
        # Start with pure noise
        y_t_nodes = torch.randn(num_total_nodes, 1, device=device)
        embedding_proj = self.embedding_projection(graph_embeddings)

        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            
            # Create time tensors
            time_tensor_per_graph = torch.full((num_graphs,), t, device=device, dtype=torch.long)
            time_embed = self.ddm.time_embedding(time_tensor_per_graph)
            conditioning_signal_per_graph = time_embed + embedding_proj
            conditioning_signal_nodes = dgl.broadcast_nodes(batched_dgl_graph, conditioning_signal_per_graph)
            
            # Predict the clean signal (x_0)
            predicted_x0_nodes, _ = self.ddm.net(
                batched_dgl_graph, 
                x_t=y_t_nodes, 
                time_embed=conditioning_signal_nodes
            )

            # DDIM reverse step
            if i > 0:
                t_prev = ddim_timesteps[i-1]
                
                # Get alpha values
                alpha_t = self.ddm.sqrt_alphas_bar[t] ** 2
                alpha_t_prev = self.ddm.sqrt_alphas_bar[t_prev] ** 2
                
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                
                # Predict epsilon (noise)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                pred_eps = (y_t_nodes - torch.sqrt(alpha_t) * predicted_x0_nodes) / sqrt_one_minus_alpha_t
                
                # DDIM update
                mean = torch.sqrt(alpha_t_prev) * predicted_x0_nodes + torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_eps
                
                if sigma_t > 0:
                    noise = torch.randn_like(y_t_nodes)
                    y_t_nodes = mean + sigma_t * noise
                else:
                    y_t_nodes = mean
            else:
                y_t_nodes = predicted_x0_nodes

        # Average node predictions to get graph-level prediction
        batched_dgl_graph.ndata['__temp_logits'] = y_t_nodes
        graph_level_logits = dgl.mean_nodes(batched_dgl_graph, '__temp_logits')
        del batched_dgl_graph.ndata['__temp_logits']
        
        return graph_level_logits.squeeze(-1)

    @torch.no_grad() 
    def sample_original(self, batched_dgl_graph, graph_embeddings):
        """ The original inference (reverse) sampling pass. """
        device = graph_embeddings.device
        num_total_nodes = batched_dgl_graph.num_nodes()
        num_graphs = graph_embeddings.shape[0]

        # Start with pure noise, one value per node in the entire batch.
        y_t_nodes = torch.randn(num_total_nodes, 1, device=device)

        embedding_proj = self.embedding_projection(graph_embeddings)

        for t in reversed(range(self.ddm.T)):
            time_tensor_per_graph = torch.full((num_graphs,), t, device=device, dtype=torch.long)
            time_embed = self.ddm.time_embedding(time_tensor_per_graph)
            conditioning_signal_per_graph = time_embed + embedding_proj
            conditioning_signal_nodes = dgl.broadcast_nodes(batched_dgl_graph, conditioning_signal_per_graph)
            
            # Predict the clean signal (x_0) at the node level.
            predicted_x0_nodes, _ = self.ddm.net(
                batched_dgl_graph, 
                x_t=y_t_nodes, 
                time_embed=conditioning_signal_nodes
            )

            # DDIM-style reverse step
            time_tensor_nodes = dgl.broadcast_nodes(batched_dgl_graph, time_tensor_per_graph)
            alpha_t_sqrt = extract(self.ddm.sqrt_alphas_bar, time_tensor_nodes, y_t_nodes.shape)
            
            if t > 0:
                alpha_t_prev_sqrt = extract(self.ddm.sqrt_alphas_bar, time_tensor_nodes - 1, y_t_nodes.shape)
            else:
                alpha_t_prev_sqrt = torch.ones_like(alpha_t_sqrt) 

            sqrt_one_minus_alpha_t_prev_sq = torch.sqrt(1.0 - alpha_t_prev_sqrt.pow(2))
            
            # Predict noise from the predicted clean state (x0)
            sqrt_one_minus_alphas_bar_t = extract(self.ddm.sqrt_one_minus_alphas_bar, time_tensor_nodes, y_t_nodes.shape)
            pred_noise = (y_t_nodes - alpha_t_sqrt * predicted_x0_nodes) / (sqrt_one_minus_alphas_bar_t + 1e-9)
            
            y_t_nodes = alpha_t_prev_sqrt * predicted_x0_nodes + sqrt_one_minus_alpha_t_prev_sq * pred_noise

        # The final denoised value is a prediction at each node. We average them to get a graph-level prediction.
        final_node_logits = y_t_nodes
        # Use a temporary feature name for mean_nodes
        batched_dgl_graph.ndata['__temp_logits'] = final_node_logits
        graph_level_logits = dgl.mean_nodes(batched_dgl_graph, '__temp_logits')
        del batched_dgl_graph.ndata['__temp_logits'] # Clean up
        
        return graph_level_logits.squeeze(-1)

def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
