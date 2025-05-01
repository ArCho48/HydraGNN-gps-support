import pickle, os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch_geometric.utils import to_networkx

dataset = pickle.load(open('graphs.pkl','rb'))
# nx_graphs = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"]) for data in dataset]
nx_graphs_sorted = sorted(dataset, key=lambda data: data.x.shape[0])
nx_graphs_sorted = [to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True) for data in nx_graphs_sorted]
top50 = nx_graphs_sorted[-50:]

# Diameters
diameters = [nx.diameter(G) for G in nx_graphs_sorted]
fig,ax = plt.subplots()
ax.hist(diameters,bins=5)
ax.set_xlabel('Diameter',fontsize=15)
ax.set_ylabel('Number of graphs',fontsize=15)
plt.savefig("diameter.png", format="PNG")

# Avg. shortest-path-lengths
path_len = [nx.average_shortest_path_length(G) for G in nx_graphs_sorted]
len(path_len)
fig,ax = plt.subplots(figsize=(8, 6))
ax.hist(path_len,bins='auto')
ax.set_xlabel('Avg. Path Length',fontsize=15)
ax.set_ylabel('Number of graphs',fontsize=15)
plt.savefig("avg_path_len.png", format="PNG")

if not os.path.exists('degrees'):
    os.makedirs('degrees')
# Degree histograms (top50)
for graph in top50:
 # Assuming you have a NetworkX graph called 'graph'
     degree_counts = nx.degree_histogram(graph)
     degrees = range(len(degree_counts))

     plt.figure(figsize=(8, 6))
     plt.bar(degrees, degree_counts)
     plt.xlabel("Degree",fontsize=15)
     plt.ylabel("Frequency",fontsize=15)
     plt.savefig("degrees/degree_"+str(graph.number_of_nodes())+".png", format="PNG")

def draw(G, pos, measures, measure_name, folder):
    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=75, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    edges = nx.draw_networkx_edges(G, pos,width=0.1,arrows=False)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.savefig(folder+'/'+folder.rstrip()+'_'+str(G.number_of_nodes())+'.png',format='PNG')

if not os.path.exists('eigens'):
    os.makedirs('eigens')
for G in top50: draw(G, nx.spring_layout(G, seed=675), nx.eigenvector_centrality(G,max_iter=500), 'Eigenvector Centrality', 'eigens')

if not os.path.exists('comms'):
    os.makedirs('comms')
for G in top50: draw(G, nx.spring_layout(G, seed=675), nx.communicability_betweenness_centrality(G), 'Communicability Centrality', 'comms')
