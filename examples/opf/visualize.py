import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import OPFDataset

data = OPFDataset('dataset/')
index = np.random.randint(len(data))
data = data[index]

graph = to_networkx(data, to_undirected=False)

# Define colors for nodes and edges
node_type_colors = {
    "generator": 'r',
    "bus": 'g',
    "shunt": 'b',
    "load": 'm'
}

node_colors = []
labels = {}
for node, attrs in graph.nodes(data=True):
    node_type = attrs["type"]
    color = node_type_colors[node_type]
    node_colors.append(color)
    if attrs["type"] == "generator":
        labels[node] = f"G{node}"
    elif attrs["type"] == "bus":
        labels[node] = f"B{node}"
    elif attrs["type"] == "shunt":
        labels[node] = f"S{node}"
    elif attrs["type"] == "load":
        labels[node] = f"L{node}"

# Define colors for the edges
edge_type_colors = {
    ('bus', 'ac_line', 'bus'): "#8B4D9E",  
    ('bus', 'ac_line', 'bus'): "#DFB825",
    ('bus', 'ac_line', 'bus'): "#70B349",
    ('bus', 'transformer', 'bus'): "#DB5C64",
    ('bus', 'transformer', 'bus'): '#ADD8E6',
    ('bus', 'transformer', 'bus'): '#808000',
    ('generator', 'generator_link', 'bus'): '#40E0D0',  
    ('bus', 'generator_link', 'generator'):  '#FAC205',
    ('load', 'load_link', 'bus'):  '#FF81C0',
    ('bus', 'load_link', 'load'):  '#650021',
    ('shunt', 'shunt_link', 'bus'):  '#FF00FF',
    ('bus', 'shunt_link', 'shunt'):  '#BBF90F'
}

edge_colors = []
for from_node, to_node, attrs in graph.edges(data=True):
    edge_type = attrs["type"]
    color = edge_type_colors[edge_type]

    graph.edges[from_node, to_node]["color"] = color
    edge_colors.append(color)


# Draw the graph
pos = nx.spring_layout(graph, k=2)
nx.draw_networkx(
    graph,
    pos=pos,
    labels=labels,
    with_labels=True,
    node_color=node_colors,
    edge_color=edge_colors,
    node_size=600,
)
plt.show()
