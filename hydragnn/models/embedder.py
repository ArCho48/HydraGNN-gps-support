import pdb
import torch
from torch import nn

class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        lpe_dim: int,
        pe_dim: int,
        ce_dim: int,
        node_in_dim: int,
        edge_in_dim: int,
        rel_pe_dim: int,
        hidden_dim: int,
        use_global_attn: bool = True,
        use_encodings: bool = True,
        use_edge_attr: bool = True,
        is_edge_model: bool = True,
    ):
        super().__init__()
        self.use_global_attn = use_global_attn
        self.use_encodings = use_encodings
        self.use_edge_attr = use_edge_attr
        self.is_edge_model = is_edge_model

        if self.use_global_attn or self.use_encodings:
            # Compute total input dims
            node_feats = []
            if self.use_global_attn:
                node_feats.append(lpe_dim)
            if node_in_dim > 0:
                node_feats.append(node_in_dim)
            if self.use_encodings:
                node_feats.append(pe_dim)
                if ce_dim > 0:
                    node_feats.append(ce_dim)
            self.node_input_dim = sum(node_feats)

            # Single linear to hidden_dim for nodes
            self.node_proj = nn.Linear(self.node_input_dim, hidden_dim, bias=False)

            if self.is_edge_model:
                edge_feats = [rel_pe_dim]
                if self.use_edge_attr and edge_in_dim > 0:
                    edge_feats.append(edge_in_dim)
                self.edge_input_dim = sum(edge_feats)
                self.edge_proj = nn.Linear(self.edge_input_dim, hidden_dim, bias=False)

    def forward(self, data, conv_args):
        if self.use_global_attn or self.use_encodings:
            # === Node features ===
            feats = []
            if hasattr(data, "x"):
                feats.append(data.x.float())              # [N, node_in_dim]
            if self.use_global_attn:
                feats.append(data.lpe)                    # [N, lpe_dim]
            if self.use_encodings:
                feats.append(data.pe)                     # [N, pe_dim]
                if hasattr(data, "ce"):
                    feats.append(data.ce)                 # [N, ce_dim]
            # concat raw and project
            x = torch.cat(feats, dim=-1)                  # [N, node_input_dim]
            x = self.node_proj(x)                         # [N, hidden_dim]

            if self.is_edge_model:
                e_feats = [data.rel_pe]                   # [E, rel_pe_dim]
                if self.use_edge_attr and "edge_attr" in conv_args:
                    e_feats.append(conv_args["edge_attr"])# [E, edge_in_dim]
                e = torch.cat(e_feats, dim=-1)            # [E, edge_input_dim]
                e = self.edge_proj(e)                     # [E, hidden_dim]
                conv_args["edge_attr"] = e

            return x, data.pos, conv_args
        else:
            return data.x, data.pos, conv_args

    # source_pe = data.pe[data.edge_index[0]]
    # target_pe = data.pe[data.edge_index[1]]
    # data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference