import os, json, pdb,pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch_geometric.utils import to_networkx
from scipy.linalg import expm
from torch_geometric.utils import to_dense_adj
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from networkx.algorithms.communicability_alg import communicability_exp

def add_high_centrality_edges(data: Data, folder: str, filename: str, method: str = 'eigen', threshold: float = 0.01, max_iter: int = 100, tol: float = 1e-6):
    """
    Given a PyG Data object (with `pos` [N×3] attribute), compute node eigenvector centralities,
    derive edge-centralities as product of node scores, then add the top `threshold` fraction of 
    non-existent edges. Plot original vs augmented graph using a PCA projection of data.pos.

    Args:
        data (Data): PyG Data with `edge_index` (undirected) and `pos` (N×3 tensor).
        threshold (float): fraction (0 < threshold < 1) of all possible new edges to add.
        max_iter (int): maximum power-iteration steps for eigen-solver.
        tol (float): convergence tolerance for eigen-solver.

    Returns:
        new_data (Data): copy of `data` with added edges.
    """
    N = data.num_nodes
    # build adjacency matrix and existing‐edge set
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=N).astype(float)
    A = (A + A.T) / 2
    rows, cols = A.nonzero()
    existing = set(zip(rows.tolist(), cols.tolist()))

    # 1a) Eigen‐based scores: for each i<j, score = c_i * c_j
    if method == 'eigen':
        vals, vecs = spla.eigs(A, k=1, which='LM', maxiter=max_iter, tol=tol)
        cent = torch.from_numpy(vecs[:,0].real)
        cent = cent / cent.sum()
        scores = [
            ((i, j), cent[i].item() * cent[j].item())
            for i in range(N) for j in range(i+1, N)
            if (i, j) not in existing
        ]

    # 1b) Communicability scores: communicability_exp(G)[i][j]
    elif method == 'communicability':
        G = to_networkx(data, to_undirected=True)
        comm = communicability_exp(G)  # dict-of-dict
        scores = [
            ((i, j), comm[i][j])
            for i in range(N) for j in range(i+1, N)
            if (i, j) not in existing
        ]

    else:
        raise ValueError("`method` must be 'eigen' or 'communicability'")

    # 2) pick top-threshold fraction
    scores.sort(key=lambda x: x[1], reverse=True)
    K = int(len(scores) * threshold)
    new_edges = [pair for pair, _ in scores[:K]]

    # 3) assemble new edge_index
    ei = data.edge_index.clone().t().tolist()
    for i, j in new_edges:
        ei.append([i, j]); ei.append([j, i])
    new_edge_index = torch.tensor(ei, dtype=torch.long).t()
    new_data = Data(
        x=data.x, edge_index=new_edge_index, pos=data.pos,
        **{k: v for k, v in data if k not in ('x','edge_index','pos')}
    )

    # 4) PCA project 3D → 2D
    pos_np = data.pos.cpu().numpy()
    centered = pos_np - pos_np.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj2d = centered.dot(vt[:2].T)
    pos2d = {i: (float(proj2d[i,0]), float(proj2d[i,1])) for i in range(N)}



    # 5) PCA project 3D pos → 2D
    pos_np = data.pos.cpu().numpy()  # shape (N,3)
    pos_centered = pos_np - pos_np.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pos_centered, full_matrices=False)
    pcs = vt[:2]  # first two principal axes
    proj2d = pos_centered.dot(pcs.T)  # shape (N,2)
    pos2d = {i: (float(proj2d[i,0]), float(proj2d[i,1])) for i in range(data.num_nodes)}

    # 6) Plot side-by-side
    G_orig = to_networkx(data, to_undirected=True)
    G_new  = to_networkx(new_data, to_undirected=True)

    threshold_pct = threshold * 100
    orig_edges = list(G_orig.edges())
    new_edges_list = new_edges  # these are undirected (i<j)

    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # --- Original Graph View ---
    nx.draw_networkx_nodes(
        G_orig, pos2d,
        node_color='lightblue', edgecolors='black', node_size=20,
        ax=axes[0]
    )
    nx.draw_networkx_edges(
        G_orig, pos2d,
        edgelist=orig_edges, style='dotted',
        edge_color='lightgrey', width=0.5,
        ax=axes[0]
    )
    axes[0].set_title("Original Graph")
    axes[0].text(
        0.01, 0.97,
        f"Threshold = {threshold_pct:.1f}%\nTotal edges = {len(orig_edges)}",
        transform=axes[0].transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    axes[0].axis('off')

    # --- Augmented Graph View ---
    nx.draw_networkx_nodes(
        G_new, pos2d,
        node_color='lightblue', edgecolors='black', node_size=20,
        ax=axes[1]
    )
    # existing edges still dotted light grey
    nx.draw_networkx_edges(
        G_new, pos2d,
        edgelist=orig_edges, style='dotted',
        edge_color='lightgrey', width=0.5,
        ax=axes[1]
    )
    # new edges as dotted red
    nx.draw_networkx_edges(
        G_new, pos2d,
        edgelist=new_edges_list, style='dotted',
        edge_color='red', width=1.0,
        ax=axes[1]
    )
    axes[1].set_title("Augmented Graph")
    axes[1].text(
        0.01, 0.97,
        f"Threshold = {threshold_pct:.1f}%\nOrig edges = {len(orig_edges)}\nNew edges = {len(new_edges_list)}",
        transform=axes[1].transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    # Legend
    legend_handles = [
        mlines.Line2D([], [], color='lightgrey', linestyle='dotted', linewidth=0.5, label='Existing edges'),
        mlines.Line2D([], [], color='red',       linestyle='dotted', linewidth=1.0, label='New edges'),
    ]
    axes[1].legend(handles=legend_handles, loc='best', fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, folder.rstrip('_') + '_' + filename + '.png'), format='PNG')
    plt.close()


# Configurable run choices (JSON file that accompanies this example script).
filename = '../tmqm.json'
with open(filename, "r") as f:
    config = json.load(f)

var_config = config["NeuralNetwork"]["Variables_of_interest"]

dataset = SimplePickleDataset(basedir='/Users/67c/Documents/github/Hydragnn-data-res-backup/tmqm/non-encoder/tmqm.pickle', label="testset", var_config=var_config)
dataset = [data for data in dataset]

nx_graphs_sorted = sorted(dataset, key=lambda data: data.x.shape[0])
# nx_graphs_sorted = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True) for data in nx_graphs_sorted]
top50 = nx_graphs_sorted[-50:]

if not os.path.exists('eig_graph'):
    os.makedirs('eig_graph')

for indx, G in enumerate(top50): 
    add_high_centrality_edges(G, 'eig_graph', str(indx)+'_'+str(G.num_nodes), method='communicability')