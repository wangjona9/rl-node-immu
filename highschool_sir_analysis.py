import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data
from scipy import sparse
from copy import deepcopy as dc
import EoN

# Custom modules
from data.graph_loader import load_highschool
from utils.eigen_drops import SV, compute_eigenval
from netshield import netshield
from rl.gnn import MyModel

# Load graph
edge_index = load_highschool()
num_nodes = edge_index.max().item() + 1
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes)
graph = Data(x=node_idx, edge_index=edge_index)
nx_graph = to_networkx(graph)

# Compute consistent layout for all drawings
pos = nx.spring_layout(nx_graph, seed=42)

# Initial eigenvalue
adj = nx.adjacency_matrix(nx_graph, dtype=float)
initial_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
initial_eigenval = initial_eigenval.item()
print(f"Initial eigenvalue (High School): {initial_eigenval:.4f}")

# Budget
sir_budget = 10

# GreedySV
ob = torch.zeros(num_nodes)
actions_greedy = []
eigenvalues_greedy = [initial_eigenval]
print("\n--- GreedySV ---")
for i in range(sir_budget):
    best_res = 0
    for j in torch.where(ob == 0)[0]:
        tmp = SV(nx_graph, ob, j.item())
        if tmp > best_res:
            best_res = tmp
            action = j.item()
    ob[action] = 1
    actions_greedy.append(action)
    eigenval = compute_eigenval(nx_graph, actions_greedy)
    ev = eigenval[0].item() if hasattr(eigenval[0], 'item') else eigenval[0]
    eigenvalues_greedy.append(ev)
    print(f"Removed {action}, Eigenvalue: {ev:.4f}")

# NetShield
nx_netshield = dc(nx_graph)
eigenvalues_netshield = [initial_eigenval]
netshield_nodes = []
print("\n--- NetShield ---")
for i in range(sir_budget):
    node_list = list(nx_netshield.nodes())
    A = nx.to_numpy_array(nx_netshield, nodelist=node_list)
    node_idx_in_A = netshield(A, k=1)[0]
    node_to_delete = node_list[node_idx_in_A]
    netshield_nodes.append(node_to_delete)
    nx_netshield.remove_node(node_to_delete)
    adj = nx.adjacency_matrix(nx_netshield, dtype=float)
    eigval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    eigenvalues_netshield.append(eigval[0])
    print(f"Removed {node_to_delete}, Eigenvalue: {eigval[0]:.4f}")

# Load trained DQN model
dqn_model = MyModel(num_nodes=num_nodes, hidden_dim=128)
dqn_model.load_state_dict(torch.load("dqn_model_highschool.pt", map_location="cpu"))
dqn_model.eval()

# DQN Selection
ob = torch.zeros(num_nodes)
dqn_nodes = []
eigenvalues_dqn = [initial_eigenval]
print("\n--- DQN ---")
for i in range(sir_budget):
    mask = (ob == 0)
    with torch.no_grad():
        q_values = dqn_model(node_idx, edge_index, ob)
        masked_q = q_values[mask]
        selected_idx = torch.argmax(masked_q).item()
        action = mask.nonzero(as_tuple=True)[0][selected_idx].item()
    ob[action] = 1
    dqn_nodes.append(action)

    G_tmp = dc(nx_graph)
    G_tmp.remove_nodes_from(dqn_nodes)
    adj = nx.adjacency_matrix(G_tmp, dtype=float)
    eigval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    eigenvalues_dqn.append(eigval[0])
    print(f"Removed {action}, Eigenvalue: {eigval[0]:.4f}")

# --- Fix visualization of the contact network ---
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(nx_graph, pos, node_size=100, node_color="lightgray", edgecolors="k")
nx.draw_networkx_nodes(nx_graph, pos, nodelist=actions_greedy, node_color="red", node_size=150, edgecolors="k", label="GreedySV")
nx.draw_networkx_edges(nx_graph, pos, width=0.5, alpha=0.6)
nx.draw_networkx_labels(nx_graph, pos, font_size=8)
plt.title("High School Contact Network (GreedySV Nodes Highlighted)")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()

# --- Less connected but connected graph ---
while True:
    sparse_graph = nx.watts_strogatz_graph(n=70, k=2, p=0.05, seed=np.random.randint(10000))
    if nx.is_connected(sparse_graph):
        break

# Compute its spectral radius
sparse_adj = nx.adjacency_matrix(sparse_graph, dtype=float)
sparse_eigenval, _ = sparse.linalg.eigsh(sparse_adj, k=1, which='LA')
sparse_eigenval = sparse_eigenval.item()
print(f"Sparse Graph Eigenvalue: {sparse_eigenval:.4f}")

# Visualize the sparse connected graph
sparse_pos = nx.spring_layout(sparse_graph, seed=42)
plt.figure(figsize=(8, 8))
nx.draw(
    sparse_graph,
    pos=sparse_pos,
    node_size=100,
    node_color="lightblue",
    edge_color="gray",
    with_labels=True,
    font_size=8
)
plt.title("Connected Sparse Graph (Lower Spectral Radius)")
plt.axis("off")
plt.tight_layout()
plt.show()

# Compare eigenvalues
plt.figure(figsize=(6, 4))
plt.bar(['High School', 'Sparse Connected'], [initial_eigenval, sparse_eigenval], color=['tab:red', 'tab:blue'])
plt.ylabel('Spectral Radius (Largest Eigenvalue)')
plt.title('Spectral Radius Comparison: HS vs Sparse Graph')
plt.tight_layout()
plt.show()

# Eigenvalue curve for each method
plt.figure(figsize=(10, 6))

# Plot each method with larger markers and thicker lines
plt.plot(range(sir_budget+1), eigenvalues_greedy, label='GreedySV', marker='o', markersize=8, linewidth=2.5, color='tab:green')
plt.plot(range(sir_budget+1), eigenvalues_netshield, label='NetShield', marker='^', markersize=8, linewidth=2.5, color='tab:orange')
plt.plot(range(sir_budget+1), eigenvalues_dqn, label='DQN', marker='s', markersize=8, linewidth=2.5, color='tab:red')


# Labels and title
plt.xlabel("Number of Nodes Removed", fontsize=12)
plt.ylabel("Largest Eigenvalue", fontsize=12)
plt.title("Eigenvalue Reduction After Node Removal", fontsize=14)

# Grid and ticks
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Enhanced legend (key)
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)

# Layout adjustment
plt.tight_layout()
plt.show()
plt.savefig("plots/eigen_reduction_comparison.png")

