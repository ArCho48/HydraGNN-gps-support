import os, json, pdb
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

from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

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

def draw_communicability_graph(data, comm_dist, folder, filename, threshold_percent=10):
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes

    # Compute threshold value
    triu_indices = np.triu_indices(num_nodes, k=1)
    all_dists = comm_dist[triu_indices]
    threshold = np.percentile(all_dists, threshold_percent)

    existing_edges = set([tuple(sorted((u, v))) for u, v in G.edges()])
    low_dist_edges = set(
        (i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)
        if comm_dist[i, j] <= threshold
    )

    solid_existing = low_dist_edges & existing_edges
    solid_new = low_dist_edges - existing_edges
    dotted_existing = existing_edges - solid_existing

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Convert node positions to numpy array for quadrant analysis
    coords = np.array([pos[n] for n in G.nodes()])
    xs, ys = coords[:, 0], coords[:, 1]

    # Define quadrants
    x_med, y_med = np.median(xs), np.median(ys)
    quadrants = {
        'upper right': ((xs > x_med) & (ys > y_med)).sum(),
        'upper left': ((xs < x_med) & (ys > y_med)).sum(),
        'lower left': ((xs < x_med) & (ys < y_med)).sum(),
        'lower right': ((xs > x_med) & (ys < y_med)).sum(),
    }

    # Choose the quadrant with the fewest nodes
    best_loc = min(quadrants, key=quadrants.get)

    # Plot
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=list(dotted_existing), style='dotted', edge_color='gray', width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_existing), style='solid', edge_color='blue', width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_new), style='solid', edge_color='red', width=0.5)

    # Legend
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color='gray', linestyle='dotted', linewidth=0.5, label='Existing edge (high comm. distance)'),
        mlines.Line2D([], [], color='blue', linestyle='solid', linewidth=0.5, label='Existing edge (low comm. distance)'),
        mlines.Line2D([], [], color='red', linestyle='solid', linewidth=0.5, label='New edge (low comm. distance)'),
    ]
    plt.legend(handles=legend_handles, loc=best_loc)
    plt.title(f"Communicability-Aware Graph (Threshold = {threshold_percent}%)")
    plt.axis('off')
    plt.tight_layout()    
    plt.savefig(folder+'/'+folder.rstrip()+'_'+filename+'.png',format='PNG')

def draw_comparison_graphs(data, comm_dist, folder, filename, threshold_percent=10):
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes

    # Threshold on communicability distance
    triu_indices = np.triu_indices(num_nodes, k=1)
    all_dists = comm_dist[triu_indices]
    threshold = np.percentile(all_dists, threshold_percent)

    existing_edges = set([tuple(sorted((u, v))) for u, v in G.edges()])
    low_dist_edges = set(
        (i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)
        if comm_dist[i, j] <= threshold
    )

    solid_existing = low_dist_edges & existing_edges
    solid_new = low_dist_edges - existing_edges
    dotted_existing = existing_edges - solid_existing

    # Shared layout
    pos = nx.spring_layout(G, seed=42)
    coords = np.array([pos[n] for n in G.nodes()])
    xs, ys = coords[:, 0], coords[:, 1]
    x_med, y_med = np.median(xs), np.median(ys)

    quadrants = {
        'upper right': ((xs > x_med) & (ys > y_med)).sum(),
        'upper left': ((xs < x_med) & (ys > y_med)).sum(),
        'lower left': ((xs < x_med) & (ys < y_med)).sum(),
        'lower right': ((xs > x_med) & (ys < y_med)).sum(),
    }
    best_loc = min(quadrants, key=quadrants.get)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # --- Figure 1: original graph with dotted and solid edges ---
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=40, ax=axes[0])
    nx.draw_networkx_edges(G, pos, edgelist=list(dotted_existing), style='dotted', edge_color='gray', width=0.5, ax=axes[0])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_existing), style='solid', edge_color='blue', width=1.0, ax=axes[0])
    axes[0].set_title("Original Graph: High vs Low Communicability")
    axes[0].axis('off')

    # --- Figure 2: only low-distance edges ---
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=40, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_existing), style='solid', edge_color='blue', width=1.0, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=list(solid_new), style='dotted', edge_color='red', width=1.0, ax=axes[1])
    axes[1].set_title("Communicability-Based View (Low Distance Edges)")
    axes[1].axis('off')

    # --- Dynamic Legend ---
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color='gray', linestyle='dotted', linewidth=1.0, label='Existing edge (high comm. distance)'),
        mlines.Line2D([], [], color='blue', linestyle='solid', linewidth=1.0, label='Existing edge (low comm. distance)'),
        mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1.0, label='New edge (low comm. distance)'),
    ]
    axes[1].legend(handles=legend_handles, loc=best_loc, fontsize=12)

    plt.tight_layout()
    plt.savefig(folder+'/'+folder.rstrip()+'_'+filename+'.png',format='PNG')

def visualize_graph_with_distances(data, dist_matrix, folder, filename):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())

    # Edge weights: inverse communicability distance
    edge_weights = []
    for u, v in G.edges():
        d = dist_matrix[u, v]
        edge_weights.append(1 / (d + 1e-6))  # avoid div by zero

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, edge_color=edge_weights, width=2.0, edge_cmap=plt.cm.plasma)
    plt.title("Graph with Edge Color ∝ 1 / Communicability Distance")
    plt.savefig(folder+'/'+folder.rstrip()+'_'+filename+'.png',format='PNG')

def plot_heatmap(dist_matrix, folder, filename,):
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, cmap='magma', square=True)
    plt.title("Communicability Distance Heatmap")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.savefig(folder+'/'+folder.rstrip()+'_'+filename+'.png',format='PNG')

# Configurable run choices (JSON file that accompanies this example script).
filename = '../tmqm.json'
with open(filename, "r") as f:
    config = json.load(f)

var_config = config["NeuralNetwork"]["Variables_of_interest"]

dataset = SimplePickleDataset(basedir='../dataset/non-encoder/tmqm.pickle', label="testset", var_config=var_config)
dataset = [data for data in dataset]

nx_graphs_sorted = sorted(dataset, key=lambda data: data.x.shape[0])
# nx_graphs_sorted = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True) for data in nx_graphs_sorted]
top50 = nx_graphs_sorted[-50:]

if not os.path.exists('comm_graph'):
    os.makedirs('comm_graph')
# if not os.path.exists('heatmap_comm'):
#     os.makedirs('heatmap_comm')

for indx, G in enumerate(top50): 
    c_d, flag = communicability_distance(G)
    # norm_dist = normalize_minmax(c_d)
    # draw_communicability_graph(G,c_d,'comm_graph', str(G.num_nodes),threshold_percent=5)
    if flag:
        draw_comparison_graphs(G,c_d,'comm_graph', str(indx)+'_'+str(G.num_nodes),threshold_percent=5)
    # visualize_graph_with_distances(G,c_d,'comm_graph', str(G.num_nodes))
    # plot_mds( c_d, 'mds_comm', str(G.num_nodes))
    # plot_heatmap( c_d, 'heatmap_comm', str(G.num_nodes))
