import pickle
import pdb
import torch_geometric
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
g = pickle.load(open('graph.pkl','rb'))

node_color = []
node_size = []
# for each node in the graph
for node in g.nodes(data=True):
    
    if node[1]['x'][0]==1.0:
        node_color.append('blue')
        node_size.append(1.0)
    
    elif node[1]['x'][0]==6.0:
        node_color.append('red')
        node_size.append(6.0)
    
    elif node[1]['x'][0]==7.0:
        node_color.append('green')
        node_size.append(7.0)
    
    elif node[1]['x'][0]==8.0:
        node_color.append('yellow')
        node_size.append(8.0)

    elif node[1]['x'][0]==14.0:
        node_color.append('orange')
        node_size.append(14.0)

    elif node[1]['x'][0]==30.0:
        node_color.append('black')
        node_size.append(30.0)

nx.draw(g,pos=nx.spring_layout(g),with_labels=False, node_size=node_size, width=0.05,node_color=node_color)
# plt.tight_layout()
plt.savefig("Graph.png", format="PNG")