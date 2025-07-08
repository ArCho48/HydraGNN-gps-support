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

from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

comm_list = []

def normalize_minmax(dist_matrix):
    min_val = dist_matrix.min()
    max_val = dist_matrix.max()
    return (dist_matrix - min_val) / (max_val - min_val + 1e-9)

def communicability_distance(data):
    """
    Compute communicability distance matrix for a PyG Data object,
    but only if the graph is fully connected.
    
    Raises an exception if the graph is disconnected.
    """
    # Check connectivity using NetworkX
    G_nx = to_networkx(data, to_undirected=True)
    if not nx.is_connected(G_nx):
        return None, False

    # Compute adjacency matrix
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].numpy().astype(np.float64)
    adj = (adj + adj.T) / 2  # Ensure symmetry

    # Matrix exponential
    G = expm(adj)

    # Compute communicability distance
    diag_G = np.diag(G)
    dist = diag_G[:, None] + diag_G[None, :] - 2 * G
    dist = np.sqrt(np.maximum(dist, 0.0))

    return dist, True

def communicability(data, alpha=1.0):
    """
    Compute communicability matrix for a PyG Data object.
    
    Parameters:
    - data: PyG Data object
    - alpha: scalar float, user-defined damping factor
    
    Returns:
    - communicability matrix (numpy.ndarray)
    - is_connected: bool indicating whether the graph was connected
    """
    # Convert to NetworkX for connectivity check
    G_nx = to_networkx(data, to_undirected=True)
    if not nx.is_connected(G_nx):
        return None, False

    # Get adjacency matrix as float64
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].numpy().astype(np.float64)
    adj = (adj + adj.T) / 2  # Ensure symmetry

    # Compute degree matrix and D^{-1/2}, D^{1/2}
    degrees = np.sum(adj, axis=1)
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(np.power(degrees, -0.5, where=degrees > 0))

    # Compute normalized adjacency: D^{-1/2} A D^{1/2}
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt

    # Compute communicability
    comm = np.exp(-alpha) * expm(norm_adj)

    return comm, True

def draw_comparison_graphs(data, comm_dist, folder, filename, threshold_percent=10, use_tsne=False):
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes

    # --- 3D to 2D Projection (PCA or t-SNE) ---
    coords_3d = data.pos.cpu().numpy() if hasattr(data.pos, 'cpu') else data.pos.numpy()
    if use_tsne:
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else:
        reducer = PCA(n_components=2)
    coords_2d = reducer.fit_transform(coords_3d)
    pos = {i: coords_2d[i] for i in range(num_nodes)}

    xs, ys = coords_2d[:, 0], coords_2d[:, 1]
    x_med, y_med = np.median(xs), np.median(ys)
    quadrants = {
        'upper right': ((xs > x_med) & (ys > y_med)).sum(),
        'upper left': ((xs < x_med) & (ys > y_med)).sum(),
        'lower left': ((xs < x_med) & (ys < y_med)).sum(),
        'lower right': ((xs > x_med) & (ys < y_med)).sum(),
    }
    best_loc = min(quadrants, key=quadrants.get)
    # pdb.set_trace()
    # # --- Communicability thresholding ---
    # triu_indices = np.triu_indices(num_nodes, k=1)
    # all_dists = comm_dist[triu_indices]
    # threshold = np.percentile(all_dists, threshold_percent)

    # existing_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
    # low_dist_edges = set(
    #     (i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)
    #     if comm_dist[i, j] <= threshold
    # )
    # # pdb.set_trace()
    # solid_existing = low_dist_edges & existing_edges
    # solid_new = low_dist_edges - existing_edges
    # dotted_existing = existing_edges - solid_existing

    # --- Communicability thresholding based on top-K communicability ---
    triu_indices = np.triu_indices(num_nodes, k=1)
    all_comms = comm_dist[triu_indices]  # comm is the communicability matrix

    # Determine threshold for top X% most communicable links
    threshold = np.percentile(all_comms, 100 - threshold_percent)  # top X% ⇒ 100 - X percentile

    # Identify edges based on communicability strength
    high_comm_edges = set(
        (i, j) for i, j in zip(*triu_indices)
        if comm_dist[i, j] >= threshold
    )
    # pdb.set_trace()
    # Edge categories
    existing_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
    solid_existing = high_comm_edges & existing_edges
    solid_new = high_comm_edges - existing_edges
    dotted_existing = existing_edges - solid_existing

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # --- Original Graph View ---
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=40, ax=axes[0])
    nx.draw_networkx_edges(G, pos, edgelist=list(dotted_existing), style='dotted', edge_color='gray', width=0.5, ax=axes[0])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_existing), style='solid', edge_color='blue', width=1.0, ax=axes[0])
    axes[0].set_title("Original Graph: High vs Low Communicability")
    axes[0].text(0.01, 0.97,
                 f"Threshold = {threshold_percent:.1f}\nBlue edges = {len(solid_existing)}\nGray edges = {len(dotted_existing)}",
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    axes[0].axis('off')

    # --- Communicability View ---
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=40, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_existing), style='solid', edge_color='blue', width=1.0, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_new), style='dotted', edge_color='red', width=1.0, ax=axes[1])
    axes[1].set_title("Communicability-Based View (Low Distance Edges)")
    axes[1].text(0.01, 0.97,
                 f"Threshold = {threshold_percent:.1f}\nBlue edges = {len(solid_existing)}\nRed edges = {len(solid_new)}",
                 transform=axes[1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    axes[1].axis('off')
    axes[1].axis('off')

    # --- Legend ---
    legend_handles = [
        mlines.Line2D([], [], color='gray', linestyle='dotted', linewidth=1.0, label='Existing edge (high comm. distance)'),
        mlines.Line2D([], [], color='blue', linestyle='solid', linewidth=1.0, label='Existing edge (low comm. distance)'),
        mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1.0, label='New edge (low comm. distance)'),
    ]
    axes[1].legend(handles=legend_handles, loc=best_loc, fontsize=12)

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

if not os.path.exists('comm_graph'):
    os.makedirs('comm_graph')
# if not os.path.exists('heatmap_comm'):
#     os.makedirs('heatmap_comm')

# for indx, G in enumerate(top50): 
#     c_d, flag = communicability(G)#communicability_distance(G)
#     # norm_dist = normalize_minmax(c_d)
#     # draw_communicability_graph(G,c_d,'comm_graph', str(G.num_nodes),threshold_percent=5)
#     if flag:
#         # comm_list.append(c_d)
#         draw_comparison_graphs(G,c_d,'comm_graph', str(indx)+'_'+str(G.num_nodes),threshold_percent=20)
#     # visualize_graph_with_distances(G,c_d,'comm_graph', str(G.num_nodes))
#     # plot_mds( c_d, 'mds_comm', str(G.num_nodes))
#     # plot_heatmap( c_d, 'heatmap_comm', str(G.num_nodes))

thresh_list = range(1,101)#[1,2,5,10,15,25]
for thresh in thresh_list: 
    c_d, flag = communicability(top50[4])#communicability_distance(top50[-1])
    # norm_dist = normalize_minmax(c_d)
    # draw_communicability_graph(G,c_d,'comm_graph', str(G.num_nodes),threshold_percent=5)
    if flag:
        # comm_list.append(c_d)
        draw_comparison_graphs(top50[4],c_d,'comm_graph', str(4)+'_'+str(top50[4].num_nodes)+"_"+str(thresh),threshold_percent=thresh)
    # visualize_graph_with_distances(G,c_d,'comm_graph', str(G.num_nodes))
    # plot_mds( c_d, 'mds_comm', str(G.num_nodes))
    # plot_heatmap( c_d, 'heatmap_comm', str(G.num_nodes))

# pickle.dump(comm_list,open('comm.pkl','wb'))

