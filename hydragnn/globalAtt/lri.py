import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def compute_long_range_dependency(model, data, node_idx, max_k=5):
    """
    Computes the norm of the Jacobian of node representations with respect to features
    of k-hop neighbors to quantify long-range dependencies.

    Parameters:
    - model: Trained GNN model.
    - data: PyTorch Geometric Data object containing the graph.
    - node_idx: Index of the target node.
    - max_k: Maximum number of hops to consider.

    Returns:
    - A dictionary mapping hop distance k to the average Jacobian norm.
    """
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    edge_index = data.edge_index
    jacobian_norms = {}

    # Forward pass to compute node representations
    h = model(x, edge_index)

    for k in range(1, max_k + 1):
        # Get the k-hop subgraph around the target node
        subset, _, _, edge_mask = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=False)
        k_hop_nodes = subset.tolist()
        k_hop_nodes.remove(node_idx)  # Exclude the target node

        if not k_hop_nodes:
            jacobian_norms[k] = 0.0
            continue

        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Compute the representation of the target node
        h_v = h[node_idx]
        h_v.backward(retain_graph=True)

        # Compute the norm of gradients w.r.t. k-hop neighbors
        grad_norms = []
        for u in k_hop_nodes:
            grad_u = x.grad[u]
            norm = torch.norm(grad_u).item()
            grad_norms.append(norm)

        # Average norm for this hop distance
        jacobian_norms[k] = sum(grad_norms) / len(grad_norms)

    return jacobian_norms

# Example usage:
# Define a simple GCN model
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Assume 'data' is a PyTorch Geometric Data object with 'x' and 'edge_index'
# and 'model' is an instance of SimpleGCN that has been trained
# node_idx = index of the node for which to compute the metric

# jacobian_metrics = compute_long_range_dependency(model, data, node_idx, max_k=5)
# print(jacobian_metrics)
