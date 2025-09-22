import torch
import torch.nn as nn
from data.graph_loader import GraphLoader
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, subgraph
from torch_geometric.transforms import LargestConnectedComponents

import networkx as nx
import scipy.sparse as sparse

# Data
device = torch.device("cuda")
graph = GraphLoader(root='data/',graph_name="Cora")
graph = graph()

mask = graph.y==1
graph.edge_index = subgraph(mask, graph.edge_index, relabel_nodes=True)[0]
graph.x = graph.x[mask]
graph.train_mask = graph.train_mask[mask]
graph.val_mask = graph.val_mask[mask]
graph.test_mask = graph.test_mask[mask]

trans = LargestConnectedComponents()
graph = trans(graph)

nx_graph = to_networkx(graph)

from copy import deepcopy as dc

adj = nx.adjacency_matrix(nx_graph, dtype=float)
eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
node_data = [0]*nx_graph.number_of_nodes()
for i in range(nx_graph.number_of_nodes()):
    G = dc(nx_graph)
    G.remove_nodes_from([i])
    adj = nx.adjacency_matrix(G, dtype=float)
    eigenval_new, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    node_data[i] = (eigenval - eigenval_new).item()

import matplotlib.pyplot as plt

# Convert nx_graph to undirected graph
undirected_graph = nx_graph.to_undirected()

# Draw the undirected graph
plt.figure(figsize=(30, 30))
pos = nx.spring_layout(undirected_graph, k=0.15)  # Position nodes using Fruchterman-Reingold force-directed algorithm

nx.draw(nx_graph, pos, with_labels=False, node_size=300, node_color='lightblue')

# Add node data as labelss
node_labels = {i: f"{data:.2f}" for i, data in enumerate(node_data)}
nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)

plt.title("Visualizing Undirected nx_graph")
plt.savefig("undirected_nx_graph_visualization.png", format="PNG")
