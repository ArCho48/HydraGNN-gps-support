import os, json, pdb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs
from torch_geometric.utils import to_networkx
import numpy as np

from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

def scipy_eigenvector_centrality(G, 
                                 k: int = 1, 
                                 tol: float = 1e-6, 
                                 max_iter: int = 100, 
                                 normalize: bool = True):
    """
    Compute eigenvector centrality of a NetworkX graph using SciPy’s ARPACK-based solvers.
    
    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The input graph. May be directed or undirected.
    k : int, default=1
        Number of eigenvalues/vectors to compute. (We only use k=1 for the principal eigenvector.)
    tol : float, default=1e-6
        Convergence tolerance for the ARPACK solver.
    max_iter : int, default=100
        Maximum number of iterations allowed for the eigen-solver.
    normalize : bool, default=True
        If True, the returned centrality vector is L1-normalized (sum to 1).
    
    Returns
    -------
    centrality_dict : dict
        A dictionary mapping each node in G to its eigenvector centrality (a float).
    """
    # 1. Build a list of nodes and an index map:
    nodes = list(G.nodes())
    n = len(nodes)
    idx = { nodes[i]: i for i in range(n) }
    
    # 2. Assemble the adjacency matrix in sparse CSR format:
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        row.append(i)
        col.append(j)
        data.append(1.0)
        if not G.is_directed():
            # For undirected graphs, add the symmetric entry
            if i != j:
                row.append(j)
                col.append(i)
                data.append(1.0)
    
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    
    # 3. Choose the appropriate ARPACK routine:
    if not G.is_directed():
        # Undirected ⇒ symmetric adjacency ⇒ use eigsh
        # which='LA' ⇒ largest algebraic eigenvalue
        vals, vecs = eigsh(A, k=k, which='LA', tol=tol, maxiter=max_iter)
        principal_vec = vecs[:, 0]
    else:
        # Directed ⇒ possibly non-symmetric ⇒ use eigs
        # which='LR' ⇒ largest real part
        vals, vecs = eigs(A, k=k, which='LR', tol=tol, maxiter=max_iter)
        principal_vec = np.real(vecs[:, 0])
    
    # 4. Normalize if requested:
    if normalize:
        # L1-normalize so that sum of centralities = 1
        norm = np.linalg.norm(principal_vec, ord=1)
        if norm != 0:
            principal_vec = principal_vec / norm
    
    # 5. Build the output dictionary: node -> centrality
    centrality_dict = { nodes[i]: float(principal_vec[i]) for i in range(n) }
    return centrality_dict

# Configurable run choices (JSON file that accompanies this example script).
filename = '../niaid.json'
with open(filename, "r") as f:
    config = json.load(f)

var_config = config["NeuralNetwork"]["Variables_of_interest"]

dataset = SimplePickleDataset(basedir='../dataset/non-encoder/niaid.pickle', label="testset", var_config=var_config)
dataset = [data for data in dataset]
# nx_graphs = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"]) for data in dataset]
nx_graphs_sorted = sorted(dataset, key=lambda data: data.x.shape[0])
nx_graphs_sorted = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True) for data in nx_graphs_sorted]

top50 = []
for gr in reversed(nx_graphs_sorted):
    if nx.is_connected(gr):
        top50.append( gr )
    if len(top50) == 50:
        break

# Diameters
diameters = []
for G in nx_graphs_sorted:
    try:
        diameters.append(nx.diameter(G)) 
    except:
        continue
fig,ax = plt.subplots()
ax.hist(diameters,bins=5)
ax.set_xlabel('Diameter',fontsize=15)
ax.set_ylabel('Number of graphs',fontsize=15)
plt.savefig("diameter.png", format="PNG")

# Avg. shortest-path-lengths
path_len = []
for G in nx_graphs_sorted:
    try:
        path_len.append(nx.average_shortest_path_length(G))
    except:
        continue
fig,ax = plt.subplots(figsize=(8, 6))
ax.hist(path_len,bins='auto')
ax.set_xlabel('Avg. Path Length',fontsize=15)
ax.set_ylabel('Number of graphs',fontsize=15)
plt.savefig("avg_path_len.png", format="PNG")

if not os.path.exists('degrees'):
    os.makedirs('degrees')
# Degree histograms (top50)
for indx, graph in enumerate(top50):
 # Assuming you have a NetworkX graph called 'graph'
     degree_counts = nx.degree_histogram(graph)
     degrees = range(len(degree_counts))

     plt.figure(figsize=(8, 6))
     plt.bar(degrees, degree_counts)
     plt.xlabel("Degree",fontsize=15)
     plt.ylabel("Frequency",fontsize=15)
     plt.savefig("degrees/degree_"+str(indx)+'_'+str(graph.number_of_nodes())+".png", format="PNG")

def draw(G, pos, measures, measure_name, folder, indx):
    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=75, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    edges = nx.draw_networkx_edges(G, pos,width=0.1,arrows=False)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.savefig(folder+'/'+folder.rstrip()+'_'+str(indx)+'_'+str(G.number_of_nodes())+'.png',format='PNG')

if not os.path.exists('eigen_centrality'):
    os.makedirs('eigen_centrality')
for indx,G in enumerate(top50): draw(G, nx.spring_layout(G, seed=675), scipy_eigenvector_centrality(G,max_iter=1000), 'Eigenvector Centrality', 'eigen_centrality', indx) #nx.eigenvector_centrality

if not os.path.exists('comm_centrality'):
    os.makedirs('comm_centrality')
for indx,G in enumerate(top50): draw(G, nx.spring_layout(G, seed=675), nx.communicability_betweenness_centrality(G), 'Communicability Centrality', 'comm_centrality', indx)
