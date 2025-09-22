from utils.other_models import topk
from data.graph_loader import load_graph
import matplotlib.pyplot as plt


data_name = 'infect'
graph, nx_graph, random_walk_pe, node_degrees, num_nodes, node_idx = load_graph(data_name, dim=128)
num_of_del = int(0.05*num_nodes)
eigenval_dict = {}
methods = ['degree', 'centrality', 'my1', 'uBCu', 'ns', 'random']
# methods = ['uBCu']
max_k = 64
for method in methods:
    eigenval_dict[method] = []
    model = topk(nx_graph, num_of_del, k=1, method = method)
    output = model.go()
    eigenval_dict[method].append(output[0])

plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(eigenval_dict[method][0], label=method)
plt.legend()
plt.xlabel(f'Num. of Del. Nodes {num_of_del}')
plt.ylabel('Eigenvalue')
plt.title(data_name)
plt.savefig(f'datasets/baselines_{data_name}.png')