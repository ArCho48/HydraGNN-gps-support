import pdb
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from scipy.sparse.linalg import eigsh
from networkx.algorithms.centrality import harmonic_centrality
from networkx.algorithms.efficiency_measures import local_efficiency

def normalize_features(X, eps=1e-10):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + eps)

def compute_topo_features(data, k_laplacian_eigenvecs=5):
    pdb.set_trace()
    G = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])

    node_list = list(G.nodes())
    node_idx_map = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # Node Features
    node_features = []

    def to_array(d):
        return np.array([d[n] for n in node_list])

    node_features.append(to_array(dict(G.degree())))
    node_features.append(to_array(nx.clustering(G)))
    node_features.append(to_array(nx.betweenness_centrality(G)))
    node_features.append(to_array(nx.closeness_centrality(G)))
    node_features.append(to_array(nx.pagerank(G)))
    node_features.append(to_array(nx.eigenvector_centrality_numpy(G)))
    node_features.append(to_array(nx.core_number(G)))
    node_features.append(to_array(nx.eccentricity(G)))
    node_features.append(to_array(harmonic_centrality(G)))
    node_features.append(np.array([
        local_efficiency(G.subgraph(G.neighbors(v)).copy()) if len(list(G.neighbors(v))) > 1 else 0.0
        for v in node_list
    ]))

    # Laplacian Eigenvectors
    L = nx.laplacian_matrix(G).astype(float)
    num_eigen = min(k_laplacian_eigenvecs, N - 2)
    vecs = np.zeros((N, 0))  # fallback if eigsh fails
    if num_eigen > 0:
        _, vecs = eigsh(L, k=num_eigen, which='SM')

    node_features.append(vecs.T)

    # Stack and normalize
    node_features = np.vstack(node_features[:-1])
    if vecs.size > 0:
        node_features = np.vstack([node_features, vecs.T])
    node_features = node_features.T  # (N, num_features)
    node_features = normalize_features(node_features)

    # Edge Features
    edge_list = [tuple(sorted(e)) for e in G.edges()]
    edge_idx_map = {e: i for i, e in enumerate(edge_list)}
    E = len(edge_list)
    edge_features = np.zeros((E, 5))  # [betweenness, jaccard, adamic-adar, PA, common neighbors]

    betweenness = nx.edge_betweenness_centrality(G)
    for e in edge_list:
        edge_features[edge_idx_map[e], 0] = betweenness.get(e, 0.0)

    for u, v, score in nx.jaccard_coefficient(G):
        e = tuple(sorted((u, v)))
        if e in edge_idx_map:
            edge_features[edge_idx_map[e], 1] = score

    for u, v, score in nx.adamic_adar_index(G):
        e = tuple(sorted((u, v)))
        if e in edge_idx_map:
            edge_features[edge_idx_map[e], 2] = score

    for u, v, score in nx.preferential_attachment(G):
        e = tuple(sorted((u, v)))
        if e in edge_idx_map:
            edge_features[edge_idx_map[e], 3] = score

    for u, v in edge_list:
        cn = len(list(nx.common_neighbors(G, u, v)))
        edge_features[edge_idx_map[(u, v)], 4] = cn

    edge_features = normalize_features(edge_features)

    return node_features, edge_features, node_list, edge_list

