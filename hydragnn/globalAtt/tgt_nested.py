# -----------------------------------------------
# Fully Runnable Batched Triplet Graph Transformer (TGT-Ag) using NestedTensor
# Subclassing torch_geometric.nn.MessagePassing for PyG compatibility
#
# This script demonstrates how to batch multiple graphs of varying sizes without padding,
# by leveraging PyTorch’s NestedTensor API (https://docs.pytorch.org/tutorials/prototype/nestedtensor.html)
# and integrates with PyTorch Geometric by having the model inherit from MessagePassing.
#
# Requirements:
#   • PyTorch ≥ 2.0 (with NestedTensor support)
#   • torch_geometric (PyG)
#
# Usage:
#   python batched_tgt_pyg.py
#
# The script will:
#   1. Generate a small toy dataset of 3 random graphs (sizes 4, 6, 5).
#   2. Batch them with a PyG DataLoader.
#   3. Instantiate BatchedTripletGraphTransformer (subclassing MessagePassing).
#   4. Perform a forward pass and print the shapes of the final embeddings.
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import NestedTensor API (PyTorch ≥2.0)
from torch import nested_tensor

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj

# ------------------------------------------------------------------------------
#  Pre‐norm Residual Wrapper (Transformer building block)
# ------------------------------------------------------------------------------
class PreNormResidual(nn.Module):
    """
    Applies LayerNorm to the input, runs a sub‐module, then adds the input (residual).
    """
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.module(self.norm(x), *args, **kwargs)


# ------------------------------------------------------------------------------
#  Single “Batched Triplet Layer” (TGT-Ag) operating on NestedTensors
# ------------------------------------------------------------------------------
class BatchedTripletLayer(nn.Module):
    """
    One layer of the Triplet Graph Transformer (Triplet Aggregation variant, TGT-Ag),
    implemented to run on NestedTensors of shape:
      • H_nt: NestedTensor [B, N_g, d_node]
      • E_nt: NestedTensor [B, N_g, N_g, d_edge]
    Outputs updated NestedTensors of the same shapes.
    """
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        num_heads: int = 8,
        num_triplet_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert dim_node % num_heads == 0, "dim_node must be divisible by num_heads"
        assert dim_edge % num_triplet_heads == 0, "dim_edge must be divisible by num_triplet_heads"

        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.num_heads = num_heads
        self.num_triplet_heads = num_triplet_heads
        self.head_dim_node = dim_node // num_heads
        self.head_dim_edge = dim_edge // num_triplet_heads  # dt

        # ----------------------------------------------------------------------
        #  (1) Edge‐Augmented Node Attention (EGT) projections
        # ----------------------------------------------------------------------
        self.q_proj = nn.Linear(dim_node, dim_node, bias=False)
        self.k_proj = nn.Linear(dim_node, dim_node, bias=False)
        self.v_proj = nn.Linear(dim_node, dim_node, bias=False)

        # From each e[i,j], predict 2*num_heads scalars: [bias_ij (H), gate_ij (H)]
        self.edge_to_bias_and_gate = nn.Linear(dim_edge, 2 * num_heads, bias=False)

        # Final projection after concatenating heads
        self.out_proj_node = nn.Linear(dim_node, dim_node, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # ----------------------------------------------------------------------
        #  (2) Node & Edge Feed‐Forward Networks (Standard Transformer FFN)
        # ----------------------------------------------------------------------
        self.ffn_node = nn.Sequential(
            nn.Linear(dim_node, dim_node * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_node * 4, dim_node),
            nn.Dropout(dropout)
        )
        self.node_ffn_block = PreNormResidual(dim_node, self.ffn_node)

        self.ffn_edge = nn.Sequential(
            nn.Linear(dim_edge, dim_edge * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_edge * 4, dim_edge),
            nn.Dropout(dropout)
        )
        self.edge_ffn_block = PreNormResidual(dim_edge, self.ffn_edge)

        # ----------------------------------------------------------------------
        #  (3) EGT edge‐update from node attention logits → dim_edge
        # ----------------------------------------------------------------------
        self.attnlogit_to_edge = nn.Linear(num_heads, dim_edge, bias=True)

        # ----------------------------------------------------------------------
        #  (4) Triplet Aggregation (TGT-Ag) projections
        #      We only need “v_in”, “gate_and_bias” from the edge embeddings.
        # ----------------------------------------------------------------------
        #  (4.1) Project e_norm → v_in_all: [B, N_g, N_g, dim_edge] → [B, N_g, N_g, (Ht*dt)]
        self.proj_v_in = nn.Linear(dim_edge, num_triplet_heads * self.head_dim_edge, bias=False)
        #  (4.2) Project e_norm → [2 * num_triplet_heads] scalars (gate_pre, bias_pre)
        self.proj_gate_and_bias = nn.Linear(dim_edge, 2 * num_triplet_heads, bias=False)
        #  (4.3) After computing o_in & o_out (shape [B, Ht, N_g, N_g, dt]), we concat
        #        → [B, N_g, N_g, 2*Ht*dt], then project → dim_edge
        self.triplet_out_proj = nn.Linear(
            2 * num_triplet_heads * self.head_dim_edge,
            dim_edge,
            bias=True
        )

        # ----------------------------------------------------------------------
        #  (5) LayerNorms for Edge channel (pre‐norm before aggregation)
        # ----------------------------------------------------------------------
        self.norm_edge_attn = nn.LayerNorm(dim_edge)
        self.norm_triplet   = nn.LayerNorm(dim_edge)

    def forward(
        self,
        H_nt: torch.Tensor,  # NestedTensor [B, N_g, dim_node]
        E_nt: torch.Tensor   # NestedTensor [B, N_g, N_g, dim_edge]
    ):
        """
        Args:
          H_nt: NestedTensor with blocks {H^(g)} of shape [N_g, dim_node]
          E_nt: NestedTensor with blocks {E^(g)} of shape [N_g, N_g, dim_edge]
        Returns:
          H_updated_nt: NestedTensor [B, N_g, dim_node]
          E_updated_nt: NestedTensor [B, N_g, N_g, dim_edge]
        """
        # ------------------------------
        #  STEP 1.  Edge‐Augmented Node Attention (EGT)
        # ------------------------------
        # 1.1  Project node embeddings → Q, K, V
        Q_nt = self.q_proj(H_nt)  # NestedTensor [B, N_g, dim_node]
        K_nt = self.k_proj(H_nt)  # NestedTensor [B, N_g, dim_node]
        V_nt = self.v_proj(H_nt)  # NestedTensor [B, N_g, dim_node]

        # 1.2  Split into multiple heads:
        #     → [B, N_g, num_heads, head_dim_node] → permute to [B, num_heads, N_g, head_dim_node]
        head_dim = self.head_dim_node
        Q_split = Q_nt.view(*(Q_nt.shape[:-1], self.num_heads, head_dim)).permute(0, 2, 1, 3)
        K_split = K_nt.view(*(K_nt.shape[:-1], self.num_heads, head_dim)).permute(0, 2, 1, 3)
        V_split = V_nt.view(*(V_nt.shape[:-1], self.num_heads, head_dim)).permute(0, 2, 1, 3)
        # Now each is NestedTensor [B, num_heads, N_g, head_dim_node]

        # 1.3  Compute scaled dot‐product logits:  [B, num_heads, N_g, N_g]
        scale = 1.0 / (head_dim ** 0.5)
        logits_nt = torch.einsum('b h i d, b h j d -> b h i j', Q_split, K_split) * scale

        # 1.4  From each E_nt (NestedTensor [B, N_g, N_g, dim_edge]), compute bias & gate
        bg_nt = self.edge_to_bias_and_gate(E_nt)  # [B, N_g, N_g, 2*H]
        bias_nt = bg_nt[..., : self.num_heads].permute(0, 3, 1, 2)       # [B, H, N_g, N_g]
        gate_nt = torch.sigmoid(bg_nt[..., self.num_heads :].permute(0, 3, 1, 2))  # [B, H, N_g, N_g]

        # 1.5  Combine: raw_logits + bias, multiply by gate, softmax over “j”-axis
        raw_logits_nt = logits_nt + bias_nt      # [B, H, N_g, N_g]
        gated_logits_nt = raw_logits_nt * gate_nt  # [B, H, N_g, N_g]
        attn_w_nt = torch.softmax(gated_logits_nt, dim=-1)  # softmax over last axis “j”
        attn_w_nt = self.attn_dropout(attn_w_nt)

        # 1.6  Attention‐weighted sum: o_h_nt[b,h,i,d] = Σ_j attn_w_nt[b,h,i,j] * V_split[b,h,j,d]
        o_h_nt = torch.einsum('b h i j, b h j d -> b h i d', attn_w_nt, V_split)  # [B, H, N_g, head_dim_node]

        # 1.7  Concatenate heads → [B, N_g, dim_node], then final linear projection
        o_cat_nt = o_h_nt.permute(0, 2, 1, 3).reshape(
            *(o_h_nt.shape[0], -1, self.dim_node)
        )  # [B, N_g, dim_node]
        H_attn_updated_nt = H_nt + self.out_proj_node(o_cat_nt)  # Residual

        # 1.8  Build E‐update from node‐attn logits (EGT edge update)
        logits_for_edge_nt = raw_logits_nt.permute(0, 2, 3, 1)      # [B, N_g, N_g, H]
        edge_from_node_attn_nt = self.attnlogit_to_edge(logits_for_edge_nt)  # [B, N_g, N_g, dim_edge]

        # ------------------------------
        #  STEP 2.  Edge‐Channel Pre‐Norm + FFN
        # ------------------------------
        E_norm_nt = self.norm_edge_attn(E_nt)                  # [B, N_g, N_g, dim_edge]
        E_agg_nt  = E_norm_nt + edge_from_node_attn_nt         # Residual‐add
        E_ffn_updated_nt = E_agg_nt + self.edge_ffn_block(E_agg_nt)  # Final edge FFN

        # ------------------------------
        #  STEP 3.  Triplet Aggregation (TGT-Ag)  (efficient O(N_g^2))
        # ------------------------------
        E_norm_trip_nt = self.norm_triplet(E_ffn_updated_nt)   # [B, N_g, N_g, dim_edge]

        # 3.1  Project to v_in_all:  [B, N_g, N_g, dim_edge] → [B, N_g, N_g, Ht*dt]
        v_proj_nt = self.proj_v_in(E_norm_trip_nt)                # [B, N_g, N_g, Ht*dt]
        dt = self.head_dim_edge
        v_split_nt = v_proj_nt.view(*(v_proj_nt.shape[:-1], self.num_triplet_heads, dt))  # [B, N_g, N_g, Ht, dt]
        v_nt2 = v_split_nt.permute(0, 3, 1, 2, 4)                 # [B, Ht, N_g, N_g, dt]

        # 3.2  Project to (gate_pre, bias_pre) → [B, N_g, N_g, 2*Ht] → split → [B, Ht, N_g, N_g]
        gb2_nt = self.proj_gate_and_bias(E_norm_trip_nt)  # [B, N_g, N_g, 2*Ht]
        gate2_nt = torch.sigmoid(gb2_nt[..., : self.num_triplet_heads].permute(0, 3, 1, 2))  # [B, Ht, N_g, N_g]
        bias2_nt = gb2_nt[..., self.num_triplet_heads :].permute(0, 3, 1, 2)                # [B, Ht, N_g, N_g]

        # 3.3  Compute inward aggregation weights: softmax over “k” + gate
        weights_ik_nt = torch.softmax(bias2_nt, dim=-1) * gate2_nt  # [B, Ht, N_g, N_g]

        # 3.4  “Inward” output: o_in[b,h,i,j,d] = Σ_k weights_ik[b,h,i,k] * v_nt2[b,h,j,k,d]
        o_in_nt = torch.einsum('b h i k, b h j k d -> b h i j d', weights_ik_nt, v_nt2)  # [B, Ht, N_g, N_g, dt]

        # 3.5  “Outward” aggregation: swap indices k↔i in bias2_nt & gate2_nt
        bias_trans_nt = bias2_nt.permute(0, 1, 3, 2)   # [B, Ht, N_g, N_g]
        gate_trans_nt = gate2_nt.permute(0, 1, 3, 2)   # [B, Ht, N_g, N_g]
        weights_out_nt = torch.softmax(bias_trans_nt, dim=-1) * gate_trans_nt  # [B, Ht, N_g, N_g]

        v_out_nt = v_nt2.permute(0, 1, 3, 2, 4)  # [B, Ht, N_g, N_g, dt]
        o_out_nt = torch.einsum('b h k i, b h k j d -> b h i j d', weights_out_nt, v_out_nt)  # [B, Ht, N_g, N_g, dt]

        # 3.6  Concatenate o_in & o_out → [B, Ht, N_g, N_g, 2*dt]
        o_pair_nt = torch.cat([o_in_nt, o_out_nt], dim=-1)  # [B, Ht, N_g, N_g, 2*dt]
        o_pair_nt = o_pair_nt.permute(0, 2, 3, 1, 4).reshape(
            *(o_pair_nt.shape[0], -1, -1, self.num_triplet_heads * 2 * dt)
        )  # [B, N_g, N_g, 2*Ht*dt]

        new_edge_from_agg_nt = self.triplet_out_proj(o_pair_nt)  # [B, N_g, N_g, dim_edge]

        # 3.7  Residual + final edge FFN
        E_updated_nt = E_ffn_updated_nt + new_edge_from_agg_nt
        E_updated_nt = E_updated_nt + self.edge_ffn_block(E_updated_nt)

        # ------------------------------
        #  STEP 4.  Node‐Channel Feed‐Forward (Pre‐Norm + FFN)
        # ------------------------------
        H_updated_nt = H_attn_updated_nt + self.node_ffn_block(H_attn_updated_nt)  # [B, N_g, dim_node]

        return H_updated_nt, E_updated_nt


# ------------------------------------------------------------------------------
#  Full Batched TGT Model (Stack of L Layers) subclassing MessagePassing
# ------------------------------------------------------------------------------
class BatchedTripletGraphTransformer(MessagePassing):
    """
    A full-stack Triplet Graph Transformer (TGT-Ag) for batching multiple graphs
    of varying sizes using NestedTensor, implemented as a subclass of
    torch_geometric.nn.MessagePassing for PyG compatibility.
    """
    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        dim_node: int = 128,
        dim_edge: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        num_triplet_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(aggr=None)  # We do not use MessagePassing’s propagate, but subclass for PyG style

        self.dim_node = dim_node
        self.dim_edge = dim_edge

        # ----------------------------------------------------------------------
        #  (1) Initial Projections
        #      Node features → dim_node
        #      Edge features (dense) → dim_edge
        # ----------------------------------------------------------------------
        self.node_in_proj = nn.Linear(in_node_dim, dim_node)
        self.edge_in_proj = nn.Linear(in_edge_dim, dim_edge)

        # ----------------------------------------------------------------------
        #  (2) Stack of BatchedTripletLayer
        # ----------------------------------------------------------------------
        self.layers = nn.ModuleList([
            BatchedTripletLayer(
                dim_node=dim_node,
                dim_edge=dim_edge,
                num_heads=num_heads,
                num_triplet_heads=num_triplet_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, batch: Batch):
        """
        batch:  torch_geometric.data.Batch of graphs, each with:
          - x: [sum(N_g), in_node_dim]
          - edge_index, edge_attr: sparse representation
        Returns:
          node_embeddings_list:  a Python list of final node‐embedding tensors,
                                one per graph (list length = B, each [N_g, dim_node]).
          edge_embeddings_list:  a list of final edge‐embedding tensors,
                                one per graph (each [N_g, N_g, dim_edge]).
        """
        # 1) Split the batch into a list of individual Data objects
        data_list = batch.to_data_list()  # list of length B

        # 2) Build a list of initial node embeddings {H^(g)} and edge embeddings {E^(g)}:
        node_list = []
        edge_list = []
        for data in data_list:
            # 2.1  Node embedding: x[g].shape = [N_g, in_node_dim]
            H_g = self.node_in_proj(data.x)  # [N_g, dim_node]
            node_list.append(H_g)

            # 2.2  Build dense adjacency of edge features: [N_g, N_g, in_edge_dim]
            N_g = data.num_nodes
            # to_dense_adj returns shape [1, N_g, N_g, in_edge_dim] if edge_attr is given
            adj_4d = to_dense_adj(data.edge_index, max_num_nodes=N_g, edge_attr=data.edge_attr)
            E_in_g = adj_4d.squeeze(0)              # [N_g, N_g, in_edge_dim]
            # Project to hidden dim:
            E_g = self.edge_in_proj(E_in_g)         # [N_g, N_g, dim_edge]
            edge_list.append(E_g)

        # 3) Convert lists to NestedTensors:
        #    H_nt: NestedTensor of shape [B, N_g, dim_node]
        #    E_nt: NestedTensor of shape [B, N_g, N_g, dim_edge]
        H_nt = nested_tensor(node_list)
        E_nt = nested_tensor(edge_list)

        # 4) Run through each BatchedTripletLayer
        for layer in self.layers:
            H_nt, E_nt = layer(H_nt, E_nt)

        # 5) At the end, return a list of final embeddings per graph
        #    NestedTensor → list of Tensors via .unbind()
        node_embeddings_list = H_nt.unbind()  # list of length B, each [N_g, dim_node]
        edge_embeddings_list = E_nt.unbind()  # list of length B, each [N_g, N_g, dim_edge]

        return node_embeddings_list, edge_embeddings_list

