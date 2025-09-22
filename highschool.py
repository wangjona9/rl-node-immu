from data.graph_loader import load_highschool
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected
import torch
from copy import deepcopy as dc
import networkx as nx
from scipy import sparse
from netshield import netshield  
import EoN
import matplotlib.pyplot as plt
import numpy as np

from data.graph_loader import load_highschool
from utils.eigen_drops import SV, compute_eigenval
from netshield import netshield

def greedy_maximize_neighbors(G, budget):
    """
    Select a subset of nodes from the NetworkX graph G such that the number of unique neighbors of the selected nodes is maximized.
    
    Parameters:
    G (nx.Graph): A NetworkX graph.
    
    Returns:
    selected_nodes (set): A set of nodes selected to maximize the number of neighbors.
    covered_neighbors (set): A set of neighbors covered by the selected nodes.
    """
    selected_nodes = set()
    covered_neighbors = set()
    
    for i in range(budget):
        best_node = None
        best_increase = 0
        
        for node in G.nodes:
            if node in selected_nodes:
                continue
            
            neighbors = set(G.neighbors(node))
            new_neighbors = neighbors - covered_neighbors - selected_nodes
            increase = len(new_neighbors)
            
            if increase > best_increase:
                best_increase = increase
                best_node = node
        
        if best_increase == 0:
            break
        
        selected_nodes.add(best_node)
        covered_neighbors.update(G.neighbors(best_node))
        covered_neighbors.add(best_node)

    return selected_nodes, covered_neighbors


def compute_eigenval(nx_graph, node_idx):
    G = dc(nx_graph)
    G.remove_nodes_from(node_idx)
    adj = nx.adjacency_matrix(G, dtype=float)
    after_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    return after_eigenval


def SV(graph, node_index, action):
    adj = nx.adjacency_matrix(graph, dtype=float).tolil()
    mask = (node_index == 1).squeeze().cpu().numpy()
    adj[mask, :] = 0
    adj[:, mask] = 0
    eigenval, eigenvec = sparse.linalg.eigsh(adj, k=1, which='LA')

    device = node_index.device
    eigenval = eigenval.item()
    adj = torch.from_numpy(adj.todense()).to(device)
    eigenvec = torch.from_numpy(eigenvec).to(device).squeeze()
    mask = torch.zeros_like(node_index).squeeze()
    mask[action] = 1

    term1 = 2 * eigenval * (mask * (eigenvec ** 2)).sum()
    masked_eigenvec = (eigenvec * mask).reshape(-1, 1)
    term2 = (adj * (masked_eigenvec @ masked_eigenvec.T)).sum()
    eigendrop = term1 - term2
    return eigendrop.item()


# Load graph
edge_index = load_highschool()
num_nodes = 70
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes)
graph = Data(x=node_idx, edge_index=edge_index)
nx_graph = to_networkx(graph)

# Shield value computation 
shield_values = torch.zeros(num_nodes)
for i in range(num_nodes):
    shield_values[i] = SV(nx_graph, torch.zeros(num_nodes), i)
torch.save(shield_values, 'shield_values.pt')

print("Radius of the graph:", nx.radius(nx_graph))

# Initial eigenvalue
adj = nx.adjacency_matrix(nx_graph, dtype=float)
eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
print(f"Initial eigenvalue: {eigenval.item():.4f}")

# GreedySV node selection
budget = 5
ob = torch.zeros(num_nodes)
actions = []
print("\n--- Greedy SV Results ---")
for i in range(budget):
    best_res = 0
    for j in torch.where(ob == 0)[0]:
        tmp = SV(nx_graph, ob, j.item())
        if best_res < tmp:
            best_res = tmp
            action = j.item()
    ob[action] = 1
    actions.append(action)
    eigenval = compute_eigenval(nx_graph, actions)
    print(f"{action} node(s) deleted. Eigenval: {eigenval[0]:.4f}")

# NetShield selection
print("\n--- NetShield Results ---")
A = nx.to_numpy_array(nx_graph)
netshield_nodes = netshield(A, k=budget)
eigenval_ns = compute_eigenval(nx_graph, netshield_nodes)
print(f"NetShield selected nodes: {netshield_nodes}")
print(f"Eigenvalue after NetShield removal: {eigenval_ns[0]:.4f}")

# Load graph
edge_index = load_highschool()
num_nodes = 70
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes)
graph = Data(x=node_idx, edge_index=edge_index)
nx_graph = to_networkx(graph)

# Initial eigenvalue
adj = nx.adjacency_matrix(nx_graph, dtype=float)
initial_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
initial_eigenval = initial_eigenval.item()
print(f"Initial eigenvalue: {initial_eigenval:.4f}")

# Budget
budget = 5

# GreedySV
ob = torch.zeros(num_nodes)
actions_greedy = []
eigenvalues_greedy = [initial_eigenval]
print("\n--- GreedySV ---")
for i in range(budget):
    best_res = 0
    for j in torch.where(ob == 0)[0]:
        tmp = SV(nx_graph, ob, j.item())
        if best_res < tmp:
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
for i in range(budget):
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

# DQN (Random)
np.random.seed(0)
nx_dqn = dc(nx_graph)
eigenvalues_dqn = [initial_eigenval]
dqn_nodes = []
print("\n--- DQN (Random) ---")
for i in range(budget):
    remaining_nodes = list(nx_dqn.nodes())
    node_to_delete = np.random.choice(remaining_nodes)
    dqn_nodes.append(node_to_delete)
    nx_dqn.remove_node(node_to_delete)
    adj = nx.adjacency_matrix(nx_dqn, dtype=float)
    eigval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    eigenvalues_dqn.append(eigval[0])
    print(f"Removed {node_to_delete}, Eigenvalue: {eigval[0]:.4f}")

# Plot eigenvalue curves
plt.figure(figsize=(10,6))
plt.plot(range(budget+1), eigenvalues_greedy, label='GreedySV', marker='o')
plt.plot(range(budget+1), eigenvalues_netshield, label='NetShield', marker='s')
plt.plot(range(budget+1), eigenvalues_dqn, label='DQN', marker='^')
plt.xlabel("Number of Nodes Removed")
plt.ylabel("Largest Eigenvalue")
plt.title("Eigenvalue Changes After Node Removals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# SIR Simulation
def run_sir(nx_graph, removed_nodes, title):
    G_sim = nx_graph.copy()
    G_sim.remove_nodes_from(removed_nodes)
    available_nodes = list(G_sim.nodes())
    if len(available_nodes) < 5:
        raise ValueError("Too few nodes left to seed infections.")
    np.random.seed(0)
    initial_infecteds = np.random.choice(available_nodes, size=5, replace=False)

    tau = 0.05
    gamma = 0.02
    t_max = 250
    num_runs = 10

    all_t, all_S, all_I, all_R = [], [], [], []
    peaks, totals = [], []

    for _ in range(num_runs):
        t, S, I, R = EoN.fast_SIR(G_sim, tau, gamma, initial_infecteds=initial_infecteds, tmax=t_max)
        all_t.append(t)
        all_S.append(S)
        all_I.append(I)
        all_R.append(R)
        peaks.append(np.max(I))
        totals.append(R[-1])

    # Interpolation
    common_t = np.linspace(0, t_max, 300)
    S_interp = np.array([np.interp(common_t, t, S) for t, S in zip(all_t, all_S)])
    I_interp = np.array([np.interp(common_t, t, I) for t, I in zip(all_t, all_I)])
    R_interp = np.array([np.interp(common_t, t, R) for t, R in zip(all_t, all_R)])

    mean_S, std_S = S_interp.mean(axis=0), S_interp.std(axis=0)
    mean_I, std_I = I_interp.mean(axis=0), I_interp.std(axis=0)
    mean_R, std_R = R_interp.mean(axis=0), R_interp.std(axis=0)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(common_t, mean_S, label='Susceptible', color='tab:blue')
    plt.fill_between(common_t, mean_S - std_S, mean_S + std_S, alpha=0.2, color='tab:blue')
    plt.plot(common_t, mean_I, label='Infected', color='tab:orange')
    plt.fill_between(common_t, mean_I - std_I, mean_I + std_I, alpha=0.2, color='tab:orange')
    plt.plot(common_t, mean_R, label='Recovered', color='tab:green')
    plt.fill_between(common_t, mean_R - std_R, mean_R + std_R, alpha=0.2, color='tab:green')

    plt.title(f"{title} (n={num_runs})")
    plt.xlabel("Time")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return peaks, totals

# Run all SIR
peaks_orig, totals_orig = run_sir(nx_graph, [], "Original")
peaks_ns, totals_ns = run_sir(nx_graph, netshield_nodes, "NetShield")
peaks_greedy, totals_greedy = run_sir(nx_graph, actions_greedy, "GreedySV")
peaks_dqn, totals_dqn = run_sir(nx_graph, dqn_nodes, "DQN")

