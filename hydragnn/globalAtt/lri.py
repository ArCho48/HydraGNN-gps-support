import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import functional as autograd_f
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class LongRangeMeasurement:
    def __init__(self, model: nn.Module, data: Data, H_max: int, device: torch.device = None):
        self.model = model
        self.data = data
        self.H_max = H_max

        # Move model and data to device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = device
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        # Ensure input features require gradients
        self.data.x.requires_grad_(True)

        # Precompute hop‐distances matrix via NetworkX
        self._compute_hop_distances()

        # J will have shape [N, C, N, F], where
        # J[v,i,u,j] = ∂ H_v_i / ∂ X_u_j
        self.compute_all_influences()

    def _compute_hop_distances(self):
        # Build a NetworkX graph from edge_index
        edge_index_cpu = self.data.edge_index.cpu().numpy()
        G_nx = nx.Graph()
        N = self.data.num_nodes
        G_nx.add_nodes_from(range(N))
        edges = edge_index_cpu.T.tolist()
        G_nx.add_edges_from(edges)

        # Initialize hop_distances with a large sentinel (H_max+1)
        self.hop_distances = torch.full((N, N), fill_value=(self.H_max + 1), dtype=torch.long)

        # For each v, do a BFS up to H_max
        for v in range(N):
            lengths = nx.single_source_shortest_path_length(G_nx, source=v, cutoff=self.H_max)
            # lengths is a dict u->dist for dist ≤ H_max
            for u, d in lengths.items():
                self.hop_distances[v, u] = min(d, self.H_max + 1)

    def compute_all_influences(self):
        model = self.model
        data = self.data

        def forward_fn(x_flat: torch.Tensor):
            N, F = data.x.shape
            x_in = x_flat.view(N, F)
            out = model(Data(x=x_in, edge_index=data.edge_index))
            # out should be [N, C]
            return out

        # Flatten input features
        x0 = data.x
        N, F = x0.shape

        # Because autograd.functional.jacobian wants the outputs and inputs as 1D, we reshape:
        #   - Let H_flat = H.view(N*C)  (1D of length N*C)
        #   - Let X_flat = X.view(N*F)  (1D of length N*F)
        H_flat = lambda x_flat: forward_fn(x_flat).view(-1)

        # Compute full Jacobian: shape (N*C, N*F)
        X_flat = x0.view(-1)
        J_full = autograd_f.jacobian(H_flat, X_flat, create_graph=False)  # ℝ^{(N*C)×(N*F)}

        # Reshape it back to [N, C, N, F]
        J_full = J_full.view(N, -1, N, F)  # but careful: first dimension is N*C grouped
        # Actually torch returns shape [N*C, N*F], so to reshape: .view(N, C, N, F)
        J_full = J_full.view(N, -1, N, F)  # Now [N, C, N, F]

        # Store
        self.J = J_full.detach()  # no grads needed beyond here

    def _influence_I(self, v: int, u: int) -> float:
        # J_block: shape [C, F]
        J_block = self.J[v, :, u, :]  # [C, F]
        return J_block.abs().sum().item()

    def compute_T_per_node(self):
        if self.J is None:
            raise RuntimeError("Must call compute_all_influences() before computing T_h(v).")

        N = self.data.num_nodes
        H = self.H_max
        T = torch.zeros((N, H + 1), dtype=torch.float, device='cpu')

        # We'll loop over v, then for each h gather all u with hop_distances[v,u]==h
        # and sum their influences
        hop_mat = self.hop_distances  # [N, N], on CPU or GPU (it’s on CPU)
        for v in range(N):
            for h in range(H + 1):
                nodes_at_h = (hop_mat[v] == h).nonzero(as_tuple=False).view(-1).tolist()
                if len(nodes_at_h) == 0:
                    continue
                # Sum I(v,u) over all u in that shell
                total_I = 0.0
                for u in nodes_at_h:
                    total_I += self._influence_I(v, u)
                T[v, h] = total_I

        return T  # [N, H+1]

    def compute_Tbar(self):
        T = self.compute_T_per_node()  # [N, H+1]
        N = T.shape[0]
        Tbar = T.sum(dim=0) / N
        return Tbar  # [H+1]

    def compute_R(self):
        T = self.compute_T_per_node()  # [N, H+1]
        N = T.shape[0]
        H = self.H_max

        # Numerator: sum_v sum_{h=0..H} h * T[v,h]
        h_vec = torch.arange(H + 1, dtype=torch.float).view(1, -1)  # shape [1, H+1]
        numerator = (T * h_vec).sum()

        # Denominator: sum_v sum_{h=0..H} T[v,h]
        denominator = T.sum()

        if denominator.item() == 0:
            return 0.0
        return (numerator / denominator).item()


