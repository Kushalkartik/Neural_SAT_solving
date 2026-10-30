"""
Asymmetric Causal Graph Neural Network (AC-GNN)
================================================
A dual-head bipartite GNN specifically designed to detect causal directionality,
latent primary input generators, and global DAG invariants from scrambled CNFs.

Architecture:
  - Input: Bipartite HeteroData (Variable nodes, Clause nodes, Signed edges)
  - Message Passing: L-layer Asymmetric Bipartite Interaction with signed literal gating
  - Node Head: Predicts P(v is Primary Generator) ∈ [0, 1] per variable
  - Graph Head: Predicts P(formula has Causal DAG) ∈ [0, 1] globally

Author: DAC-203 Project 14 (NeuroDAG-CDCL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class AsymmetricBipartiteConv(MessagePassing):
    """
    Message passing layer between Variables and Clauses with explicit
    signed literal features and polarity-split aggregation.
    """
    def __init__(self, hidden_dim):
        super(AsymmetricBipartiteConv, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim

        # Variable -> Clause message MLP: [h_v (d) + edge_sign (1)] -> d
        self.msg_v2c = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Clause -> Variable message MLP: [h_c (d) + edge_sign (1)] -> d
        self.msg_c2v = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Clause update MLP: takes old h_c and aggregated messages -> new h_c
        self.update_c = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # Variable update MLP: takes old h_v and split pos/neg aggregated messages -> new h_v
        # Input: old h_v (d) + pos_agg (d) + neg_agg (d) = 3*d
        self.update_v = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, h_var, h_clause, edge_index_v2c, edge_attr_v2c, edge_index_c2v, edge_attr_c2v):
        # 1. Update Clause states from Variable messages
        v_idx, c_idx = edge_index_v2c[0], edge_index_v2c[1]
        msg_input_v2c = torch.cat([h_var[v_idx], edge_attr_v2c], dim=-1)
        messages_v2c = self.msg_v2c(msg_input_v2c)

        # Aggregate into clause nodes
        clause_agg = torch.zeros_like(h_clause)
        clause_agg.index_add_(0, c_idx, messages_v2c)
        h_clause_new = self.update_c(torch.cat([h_clause, clause_agg], dim=-1))

        # 2. Update Variable states from Clause messages (split by positive/negative polarity)
        c_src, v_dst = edge_index_c2v[0], edge_index_c2v[1]
        msg_input_c2v = torch.cat([h_clause_new[c_src], edge_attr_c2v], dim=-1)
        messages_c2v = self.msg_c2v(msg_input_c2v)

        # Split into positive and negative polarity streams
        pos_mask = (edge_attr_c2v.squeeze(-1) > 0).unsqueeze(-1)
        neg_mask = ~pos_mask

        pos_msgs = messages_c2v * pos_mask.float()
        neg_msgs = messages_c2v * neg_mask.float()

        pos_agg = torch.zeros_like(h_var)
        neg_agg = torch.zeros_like(h_var)

        pos_agg.index_add_(0, v_dst, pos_msgs)
        neg_agg.index_add_(0, v_dst, neg_msgs)

        var_combined = torch.cat([h_var, pos_agg, neg_agg], dim=-1)
        h_var_new = self.update_v(var_combined)

        return h_var_new, h_clause_new


class AsymmetricCausalGNN(nn.Module):
    """
    Complete Asymmetric Causal GNN model.
    Predicts node-level primary generator probabilities and global DAG probability.
    """
    def __init__(self, in_var_dim=7, in_clause_dim=1, hidden_dim=64, num_layers=4):
        super(AsymmetricCausalGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial Feature Embeddings
        self.var_embed = nn.Sequential(
            nn.Linear(in_var_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        self.clause_embed = nn.Sequential(
            nn.Linear(in_clause_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # Stack of Asymmetric Interaction Layers
        self.layers = nn.ModuleList([
            AsymmetricBipartiteConv(hidden_dim) for _ in range(num_layers)
        ])

        # Head 1: Node-Level Primary Generator Predictor
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Head 2: Global Graph-Level Causal Invariant Predictor (DAG-Entropy Gate)
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch_data):
        """
        Forward pass on PyG HeteroData or batched HeteroData.
        Returns:
            node_logits: [Total_Vars, 1]
            graph_logits: [Batch_Size, 1]
        """
        h_var = self.var_embed(batch_data['var'].x)
        h_clause = self.clause_embed(batch_data['clause'].x)

        edge_index_v2c = batch_data['var', 'in', 'clause'].edge_index
        edge_attr_v2c = batch_data['var', 'in', 'clause'].edge_attr

        edge_index_c2v = batch_data['clause', 'has', 'var'].edge_index
        edge_attr_c2v = batch_data['clause', 'has', 'var'].edge_attr

        # Multi-layer Message Passing
        for layer in self.layers:
            h_var, h_clause = layer(
                h_var, h_clause,
                edge_index_v2c, edge_attr_v2c,
                edge_index_c2v, edge_attr_c2v
            )

        # Node-level prediction
        node_logits = self.node_classifier(h_var).squeeze(-1)

        # Global graph-level pooling
        # If batched, use batch vector; otherwise use global mean/std
        if hasattr(batch_data['var'], 'batch') and batch_data['var'].batch is not None:
            batch_vec = batch_data['var'].batch
            from torch_geometric.nn import global_mean_pool
            mean_pool = global_mean_pool(h_var, batch_vec)
            
            # std pool
            var_mean_expanded = mean_pool[batch_vec]
            sq_diff = (h_var - var_mean_expanded) ** 2
            from torch_geometric.nn import global_add_pool
            counts = global_add_pool(torch.ones_like(batch_vec, dtype=torch.float32).unsqueeze(-1), batch_vec)
            std_pool = torch.sqrt(global_add_pool(sq_diff, batch_vec) / torch.clamp(counts, min=1.0) + 1e-6)
            graph_features = torch.cat([mean_pool, std_pool], dim=-1)
        else:
            mean_pool = h_var.mean(dim=0, keepdim=True)
            std_pool = h_var.std(dim=0, keepdim=True)
            graph_features = torch.cat([mean_pool, std_pool], dim=-1)

        graph_logits = self.graph_classifier(graph_features).squeeze(-1)

        return node_logits, graph_logits
