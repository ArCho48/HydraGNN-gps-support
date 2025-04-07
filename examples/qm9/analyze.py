import os
import sys
import pdb

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch_geometric
from torch_geometric.utils import to_networkx

dataset = torch_geometric.datasets.QM9(root="dataset/qm9",)


for graph in dataset:
    G_directed = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
    G = nx.to_undirected(G_directed)

    # Containers
    node_feats, edge_feats, graph_feats = [], [], []

    ## Node level encodings
    trias = nx.triangles(G)
    cliques = nx.number_of_cliques(G)
    cores = nx.core_number(G)
    clus_coeffs = nx.clustering(G)
    sq_clus = nx.square_clustering(G)
    degree = nx.degree(G)
    # node_feats.append(list(tria.values()))
    
    deg_cen = nx.degree_centrality(G)
    har_cen = nx.harmonic_centrality(G)
    sub_cen = nx.subgraph_centrality(G)
    flow_cen = nx.current_flow_closeness_centrality(G)
    comm_cen = nx.communicability_betweenness_centrality(G)
    eig_cen = nx.eigenvector_centrality_numpy(G)
    katz_cen = nx.katz_centrality_numpy(G)
    bet_cen = nx.betweenness_centrality(G)

    pdb.set_trace()
    
    ## Edge level encodings
    edge_betweenness_centrality
    edge_bet_cen = nx.edge_current_flow_betweenness_centrality(G)

    # Graph level encodings
    girth