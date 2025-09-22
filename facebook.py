import matplotlib.pyplot as plt
from data.graph_loader import load_graph
from utils.eigen_drops import SV, compute_eigenval
from data.graph_loader import load_facebook
import torch
import networkx as nx
from scipy import sparse
import numpy as np
import EoN
from netshield import netshield
from copy import deepcopy as dc
from rl.gnn import MyModel

# Load the graph
graph, nx_graph, random_walk_pe, node_degrees, num_nodes, node_idx = load_graph('infect', dim=128)
dqn_model = MyModel(num_nodes=num_nodes, hidden_dim=128)
dqn_model.load_state_dict(torch.load("dqn_model_facebook.pt", map_location="cpu"))
dqn_model.eval()

print("Radius of the graph:", nx.radius(nx_graph))

# Compute initial eigenvalue
adj = nx.adjacency_matrix(nx_graph, dtype=float)
initial_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
initial_eigenval = initial_eigenval.item()
print(f"Initial eigenvalue: {initial_eigenval:.4f}")

# SIR simulation function
def facebook_sir_multiple_runs(nx_graph, removed_nodes, title="SIR Simulation", num_runs=10):
    tau = 0.05
    gamma = 0.02
    t_max = 250

    G_sim = nx_graph.copy()
    G_sim.remove_nodes_from(removed_nodes)

    available_nodes = list(G_sim.nodes())
    if len(available_nodes) < 5:
        raise ValueError("Too few nodes left to seed infections.")
    np.random.seed(0)
    initial_infecteds_base = np.random.choice(available_nodes, size=5, replace=False)

    all_t, all_S, all_I, all_R = [], [], [], []
    peak_infections = []
    final_recovered = []

    for i in range(num_runs):
        t, S, I, R = EoN.fast_SIR(
            G_sim,
            tau,
            gamma,
            initial_infecteds=initial_infecteds_base,
            tmax=t_max
        )
        all_t.append(t)
        all_S.append(S)
        all_I.append(I)
        all_R.append(R)
        peak_infections.append(np.max(I))
        final_recovered.append(R[-1])

    return {
        "peak_infections": peak_infections,
        "final_recovered": final_recovered
    }

# Helper: compute eigenvalue reduction curve
def track_eigen_reduction(nx_graph, nodes_to_remove):
    eigenvals = []
    G = nx_graph.copy()
    current_nodes = sorted(G.nodes())  # fixed order of nodes initially

    for i in range(len(nodes_to_remove) + 1):
        # Use current nodes sorted for adjacency matrix
        current_nodes = sorted(G.nodes())
        adj = nx.adjacency_matrix(G, nodelist=current_nodes, dtype=float)
        val, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
        eigenvals.append(val.item())
        if i < len(nodes_to_remove):
            G.remove_node(nodes_to_remove[i])

    return eigenvals


# Budgets to evaluate
budgets = [0.1, 0.2]
methods = ['Original', 'NetShield', 'DQN', 'GreedySV']

# Store results
peak_means = {}
peak_stds = {}
total_means = {}
total_stds = {}

for budget_frac in budgets:
    budget = int(budget_frac * num_nodes)
    print(f"\nStarting budget of {int(budget_frac * 100)}% ({budget})")

    # Greedy SV node selection
    ob = torch.zeros(num_nodes)
    actions_greedy = []
    for i in range(budget):
        best_res = 0
        for j in torch.where(ob == 0)[0]:
            tmp = SV(nx_graph, ob, j.item())
            if best_res < tmp:
                best_res = tmp
                action = j.item()
        ob[action] = 1
        actions_greedy.append(action)

    # NetShield node selection
    nx_netshield = dc(nx_graph)
    netshield_nodes = []
    for i in range(budget):
        node_list = list(nx_netshield.nodes())
        A = nx.to_numpy_array(nx_netshield, nodelist=node_list)
        node_idx_in_A = netshield(A, k=1)[0]
        node_to_delete = node_list[node_idx_in_A]
        netshield_nodes.append(node_to_delete)
        nx_netshield.remove_node(node_to_delete)

    # Random DQN node selection (placeholder)
    # Actual DQN node selection
    ob = torch.zeros(num_nodes)
    dqn_nodes = []
    
    with torch.no_grad():
        for _ in range(budget):
            q_values = dqn_model(node_idx, graph.edge_index, ob)
            mask = ob == 0
            masked_q = q_values[mask]
            selected_idx = torch.argmax(masked_q).item()
            action = mask.nonzero(as_tuple=True)[0][selected_idx].item()
            ob[action] = 1
            dqn_nodes.append(action)


    # Run SIR simulations
    stats_original = facebook_sir_multiple_runs(nx_graph, removed_nodes=[], title=f"SIR Original (Budget={budget})")
    stats_netshield = facebook_sir_multiple_runs(nx_graph, removed_nodes=netshield_nodes, title=f"SIR NetShield (Budget={budget})")
    stats_dqn = facebook_sir_multiple_runs(nx_graph, removed_nodes=dqn_nodes, title=f"SIR DQN (Budget={budget})")
    stats_greedy = facebook_sir_multiple_runs(nx_graph, removed_nodes=actions_greedy, title=f"SIR GreedySV (Budget={budget})")

    # Store results
    peak_means[budget_frac] = [
        np.mean(stats_original["peak_infections"]),
        np.mean(stats_netshield["peak_infections"]),
        np.mean(stats_dqn["peak_infections"]),
        np.mean(stats_greedy["peak_infections"])
    ]
    peak_stds[budget_frac] = [
        np.std(stats_original["peak_infections"]),
        np.std(stats_netshield["peak_infections"]),
        np.std(stats_dqn["peak_infections"]),
        np.std(stats_greedy["peak_infections"])
    ]
    total_means[budget_frac] = [
        np.mean(stats_original["final_recovered"]),
        np.mean(stats_netshield["final_recovered"]),
        np.mean(stats_dqn["final_recovered"]),
        np.mean(stats_greedy["final_recovered"])
    ]
    total_stds[budget_frac] = [
        np.std(stats_original["final_recovered"]),
        np.std(stats_netshield["final_recovered"]),
        np.std(stats_dqn["final_recovered"]),
        np.std(stats_greedy["final_recovered"])
    ]

    # Eigenvalue reduction tracking
    eigenvals_greedy = track_eigen_reduction(nx_graph, actions_greedy)
    eigenvals_netshield = track_eigen_reduction(nx_graph, netshield_nodes)
    eigenvals_dqn = track_eigen_reduction(nx_graph, dqn_nodes)

    # Plot Eigenvalue Reduction
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(eigenvals_greedy)), eigenvals_greedy, label='GreedySV', marker='o', color='tab:green')
    plt.plot(range(len(eigenvals_netshield)), eigenvals_netshield, label='NetShield', marker='^', color='tab:orange')
    plt.plot(range(len(eigenvals_dqn)), eigenvals_dqn, label='DQN', marker='s', color='tab:red')
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Largest Eigenvalue")
    plt.title("Eigenvalue Reduction After Node Removal")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

# Swap display order: Original, NetShield, GreedySV, DQN
display_order = [0, 1, 3, 2]  
trend_order = display_order[1:]  # Exclude 'Original'
colors = ['tab:orange', 'tab:green', 'tab:red']

x = np.arange(len(budgets))
width = 0.2

# Plot grouped barplot: Peak Infections
fig, ax = plt.subplots(figsize=(10,6))
for display_i, method_idx in enumerate(display_order):
    method = methods[method_idx]
    means = [peak_means[b][method_idx] for b in budgets]
    stds = [peak_stds[b][method_idx] for b in budgets]
    ax.bar(x + display_i*width, means, width, yerr=stds, capsize=4, label=method)

for i, method_idx in enumerate(trend_order):
    method = methods[method_idx]
    means = [peak_means[b][method_idx] for b in budgets]
    ax.plot(x + 1.5*width, means, marker='o', linestyle='--', color=colors[i], label=f"{method} Trend")

ax.set_xlabel('Fraction of Nodes Removed')
ax.set_ylabel('Peak Number of Infected Nodes')
ax.set_title('Peak Infections Across Methods and Budgets')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.show()
plt.savefig("plots/fb_peak_infections.png")

# Plot grouped barplot: Total Infections
fig, ax = plt.subplots(figsize=(10,6))
for display_i, method_idx in enumerate(display_order):
    method = methods[method_idx]
    means = [total_means[b][method_idx] for b in budgets]
    stds = [total_stds[b][method_idx] for b in budgets]
    ax.bar(x + display_i*width, means, width, yerr=stds, capsize=4, label=method)

for i, method_idx in enumerate(trend_order):
    method = methods[method_idx]
    means = [total_means[b][method_idx] for b in budgets]
    ax.plot(x + 1.5*width, means, marker='o', linestyle='--', color=colors[i], label=f"{method} Trend")

ax.set_xlabel('Fraction of Nodes Removed')
ax.set_ylabel('Total Number of Infected Nodes (Recovered)')
ax.set_title('Total Infections Across Methods and Budgets')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.show()
plt.savefig("plots/fb_total_infections.png")

# Print summary table
for b in budgets:
    print(f"\nBudget {int(b*100)}%:")
    for method, mean_p, std_p, mean_t, std_t in zip(
        methods,
        peak_means[b], peak_stds[b],
        total_means[b], total_stds[b]
    ):
        print(f"{method:10s} Peak={mean_p:.2f}±{std_p:.2f}, Total={mean_t:.2f}±{std_t:.2f}")
