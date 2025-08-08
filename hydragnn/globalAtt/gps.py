##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################


import inspect
from typing import Any, Dict, Optional
import pdb
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential, LazyLinear

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch, to_dense_adj

# def make_high_centrality_edge_mask_from_eig(
#     edge_index, batch, node_eig,
#     L, keep_pct=0.10):
#     c_batch, pad_mask = to_dense_batch(node_eig, batch, max_num_nodes=L)
#     c_batch = torch.exp(c_batch)
#     cent_prod = c_batch.unsqueeze(2) * c_batch.unsqueeze(1)  
#     adj = to_dense_adj(edge_index, batch, max_num_nodes=L).squeeze(dim=0)
#     edge_bool = adj.bool()
#     # idx = torch.arange(L, device=edge_bool.device)
#     # edge_bool[:, idx, idx] = True             # block self‐loops
#     cent_prod = cent_prod.masked_fill(edge_bool, 0.0)
#     valid_row = pad_mask.unsqueeze(2)           
#     valid_col = pad_mask.unsqueeze(1)           
#     valid_both = valid_row & valid_col          
#     cent_prod = cent_prod.masked_fill(~valid_both, 0.0)
#     flat = cent_prod.flatten(1)
#     thresh = torch.quantile(flat, 1.0 - keep_pct, dim=1, keepdim=True)
#     thresh = thresh.unsqueeze(-1)
#     new_edge_bool = cent_prod >= thresh

#     return new_edge_bool

class GPSConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == "multihead":
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == "performer":
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f"{attn_type} is not supported")

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = "batch" in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        inv_node_feat: Tensor,
        equiv_node_feat: Tensor,
        **kwargs,
    ) -> Tensor:
        # Extract keyword args for transformer
        graph_batch = kwargs.get('batch', None)
        edge_index = kwargs.get('edge_index', None)
        edge_index = torch.reshape(edge_index, [2,-1])
        # node_eig = kwargs.get('eig_cent', None)

        # Local MPNN 
        hs = []
        if self.conv is not None:  
            h, equiv_node_feat = self.conv(
                inv_node_feat=inv_node_feat, equiv_node_feat=equiv_node_feat, **kwargs
            )
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + inv_node_feat
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=graph_batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Key padding mask
        h, padding_mask = to_dense_batch(inv_node_feat, graph_batch)

        # Attention mask
        # adj = to_dense_adj(edge_index, batch=graph_batch, max_num_nodes=h.shape[1]).to(h.device).bool()
        # eig_mask = ~make_high_centrality_edge_mask_from_eig(edge_index=edge_index, node_eig=node_eig, batch=graph_batch, L=h.shape[1])
        # attention_mask = adj.bool() | eig_mask # allowed edges ∪ new edges
        # idx = torch.arange(h.shape[1], device=h.device)
        # attention_mask[:, idx, idx] = True
        # attention_mask = ~attention_mask
        # attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        # attention_mask = adj.repeat_interleave(self.heads, dim=0)

        # Global-Attention Transformer
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            # h, _ = self.attn(h, h, h, key_padding_mask=~padding_mask, attn_mask=attention_mask, need_weights=False)
            h, _ = self.attn(h, h, h, key_padding_mask=~padding_mask, need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=padding_mask)

        h = h[padding_mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + inv_node_feat  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=graph_batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=graph_batch)
            else:
                out = self.norm3(out)

        return out, equiv_node_feat

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads}, "
            f"attn_type={self.attn_type})"
        )
