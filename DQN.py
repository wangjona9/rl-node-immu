import torch
from torch_geometric.utils import to_networkx, to_undirected, degree
from torch_geometric.data import Data
import numpy as np
import random
from dqn_env import *

import networkx as nx
import scipy.sparse as sparse
from copy import deepcopy as dc
from torch_geometric.loader import DataLoader
import argparse

from rl.gnn import MyModel, RGCN
#from data.graph_loader import load_highschool
from data.graph_loader import load_facebook 

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


def parse_args():
    parser = argparse.ArgumentParser(description='Node Immunization with RL')
    parser.add_argument('--device', type=str, default='0', help='Device to use for computation')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--budget', type=int, default=5, help='Budget for node immunization')
    parser.add_argument('--max_episodes', type=int, default=10000, help='Maximum number of episodes')
    parser.add_argument('--max_epsilon', type=float, default=1.0, help='Starting value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--epsilon_decay', type=float, default=1/1600, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter for prioritized replay buffer')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for prioritized replay buffer')
    parser.add_argument('--update_target_iters', type=int, default=1000, help='Interval for updating target network')
    parser.add_argument('--buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--data', type=str, default='infect', help='Dataset to use for training')
    parser.add_argument('--reward', type=str, default='shield', help='Reward function to use for training')
    return parser.parse_args()

args = parse_args()

# Set the random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

############################ DATA LOADING #########################
device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')
edge_index = load_facebook()
num_nodes = edge_index.max().item() + 1
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes)
graph = Data(x=node_idx, edge_index=edge_index)
nx_graph = to_networkx(graph)
graph, node_idx = graph.to(device), node_idx.to(device)


############################ MODEL #########################
hidden_dim = args.hidden_dim
lr = args.lr
q_value = MyModel(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
#q_value.embeddings.weight.data = random_walk_pe.to(device)
target_q_value = MyModel(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
target_q_value.load_state_dict(q_value.state_dict())
optimizer = torch.optim.AdamW(q_value.parameters(), lr=lr, weight_decay=1e-4)


############################ MAIN TRAINING LOOP #########################
discount_rate = 1
max_episodes = args.max_episodes
max_epsilon = args.max_epsilon
min_epsilon = args.min_epsilon
epsilon_decay = args.epsilon_decay
epsilon = args.max_epsilon
batch_size = args.batch_size
max_grad_norm = args.max_grad_norm
update_target_iters = args.update_target_iters
budget = int(0.05*num_nodes)
verbose = args.verbose
beta = args.beta
alpha = args.alpha

shield_values = torch.load("datasets/shield_values_highschool.pt").to(device)


env = NodeImmunization(budget=budget, device=device, graph=nx_graph, reward=args.reward)
# buffer = ReplayBuffer(num_nodes, size=args.buffer_size, batch_size=batch_size, gamma=discount_rate)
buffer = PrioritizedReplayBuffer(num_nodes, size=args.buffer_size, batch_size=batch_size, alpha=alpha)


step_count = 0

eigen_vals = []
total_loss = []
neighbors = []

num_iters = 0

import csv
with open(args.reward + '_iter_shield_value.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'ShieldValue', 'Action'])  # Write the header
    
    for episode in range(max_episodes):

        ob = env.register(graph)
        for t in range(budget):
            mask = (env.x == 0).squeeze()
            if random.random() < epsilon:
                randomly_selected_node = random.choice(range(mask.sum().item()))
                index = mask.nonzero().squeeze()[randomly_selected_node]
                action = index.item()
                
                # sample_prob = shield_values / shield_values[mask].sum()
                # selected_node = torch.multinomial(sample_prob[mask], 1).item()
                # index = mask.nonzero().squeeze()[selected_node]
                # action = index.item()
            else:
                if q_value.training:
                    q_value.reset_noise()
                    target_q_value.reset_noise()
                    with torch.no_grad():
                        q_values = q_value(node_idx, graph.edge_index, ob)
                        q_values_masked = q_values[mask]  # only valid actions
                        selected_node = torch.argmax(q_values_masked).item()
                        masked_indices = mask.nonzero(as_tuple=False).squeeze()
                        index = masked_indices[selected_node]


            next_ob, reward, done = env.step(action)
            buffer.store(ob.cpu().squeeze().numpy(), action, reward, next_ob.cpu().squeeze().numpy(), int(done))
            ob = next_ob
            writer.writerow([num_iters + 1, reward, action])  # Write the data


            output_ready = False
            if len(buffer) >= 2*batch_size:
                # batch_data = buffer.sample_batch()
                batch_data = buffer.sample_batch(beta)
                batch_dataset = []

                for i in range(batch_data["obs"].shape[0]):
                    G = dc(graph).cpu()
                    G.x = node_idx.cpu()
                    next_obs = torch.from_numpy(batch_data['next_obs'][i])
                    obs = torch.from_numpy(batch_data['obs'][i])
                    G.obs = obs
                    G.next_obs = next_obs
                    G.rews = torch.tensor([batch_data['rews'][i]])
                    G.y = torch.zeros(num_nodes, 1).squeeze()
                    G.y[int(batch_data['acts'][i])] = 1
                    G.done = torch.tensor([bool(batch_data['done'][i])], dtype=torch.bool)
                    G.indices = batch_data["indices"][i]
                    G.weights = batch_data["weights"][i]
            
                    batch_dataset.append(G)

                loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)
                for batch in loader:
                    batch = batch.to(device)
                    # Target

                    with torch.no_grad():
                        from torch_geometric.nn import global_mean_pool

                        # Forward pass on next_obs
                        next_q_values = target_q_value(batch.x, batch.edge_index, batch.next_obs)  # [total_nodes, num_actions]

                        # Get greedy actions from main network
                        greedy_next_q = q_value(batch.x, batch.edge_index, batch.next_obs)
                        pooled_greedy_next_q = global_mean_pool(greedy_next_q, batch.batch)
                        next_actions = pooled_greedy_next_q.argmax(dim=1)

                        # Evaluate those actions using target network
                        pooled_next_q = global_mean_pool(next_q_values, batch.batch)
                        target_q_values = pooled_next_q.gather(1, next_actions.unsqueeze(1)).squeeze()


                        # print("batch.rews:", batch.rews.shape)
                        # print("target_q_values:", target_q_values.shape)

                        target = batch.rews + discount_rate * target_q_values
                        target[batch.done] = batch.rews[batch.done]



                q_values = q_value(batch.x, batch.edge_index, batch.obs)
                q_values = q_values[batch.y==1].squeeze()

                elementwise_loss = torch.nn.functional.smooth_l1_loss(q_values, target, reduction='none')

                weights = torch.tensor(batch.weights, dtype=torch.float32, device=device)
                loss = torch.mean(elementwise_loss * weights)

                
                buffer.update_priorities(batch.indices.tolist(), elementwise_loss.detach().cpu().numpy() + buffer.eps)

                # if loss.item()>10:
                #     import pdb; pdb.set_trace()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_value.parameters(), max_grad_norm)
                optimizer.step()
                output_ready = True

                num_iters += 1
                if num_iters % update_target_iters == 0:
                    target_q_value.load_state_dict(q_value.state_dict())

        if verbose and output_ready:
            G = dc(nx_graph)
            G.remove_nodes_from(torch.where(ob == 1)[0].cpu().numpy())
            adj = nx.adjacency_matrix(G, dtype=float)
            eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
            eigen_vals.append(eigenval.item())
            total_loss.append(elementwise_loss.mean().item())
            num_neighbors = compute_total_degree(nx_graph, torch.where(ob == 1)[0].cpu().numpy())
            neighbors.append(num_neighbors)
            

            print("Iteration: {} , Avg Loss: {:.4f}, Eigenval: {:.4f}, Neighbor: {:.4f}, Mean: {:.4f}, STD: {:.4f}".format(
            num_iters, 
            total_loss[-1], 
            eigen_vals[-1], 
            neighbors[-1],
            next_q_values.detach().mean().item(),
            next_q_values.detach().std().item()
            )
            )

        
        
        
        
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        # fraction = min(episode / max_episodes, 1.0)
        # beta = beta + fraction * (1.0 - beta)
        beta = min(1.0, args.beta + episode * (1.0 - args.beta) / args.max_episodes)

torch.save(q_value.state_dict(), "datasets/dqn_model_facebook.pt")
#torch.save(q_value.state_dict(), "datasets/dqn_model_facebook.pt")
print("DQN saved to datasets/dqn_model_highschool.pt & dqn_model_facebook.pt")


# Loss curves
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Make sure lists are numpy arrays
eigen_vals_arr = np.array(eigen_vals)
total_loss_arr = np.array(total_loss)

# Compute correlation
#print("total_loss_arr:", total_loss_arr)
#print("eigen_vals_arr:", eigen_vals_arr)

if len(total_loss_arr) >= 2 and len(eigen_vals_arr) >= 2:
    corr, p_value = pearsonr(total_loss_arr, eigen_vals_arr)
    print(f"Pearson correlation: {corr:.4f}, p-value: {p_value:.4f}")
    plt.plot(eigen_vals_arr, total_loss_arr, marker='o')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Eigenvalue (Correlation = {corr:.2f}, p = {p_value:.2e})")
    plt.savefig("plots/loss_vs_eigenval.png")
    plt.clf()
else:
    print("Not enough data to compute Pearson correlation.")
    plt.plot(eigen_vals_arr, total_loss_arr, marker='o')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Loss")
    plt.title("Loss vs. Eigenvalue")
    plt.savefig("plots/loss_vs_eigenval_insufficient.png")
    plt.clf()


################################### TESTING ###################################

q_value.eval()
ob = env.register(graph)
done = False
mean_loss = []
while not done:
    # Run policy to collect data
    with torch.no_grad():
        mask = (env.x == 0).squeeze()
        node_value = q_value(node_idx, graph.edge_index, ob)
        selected_node = torch.argmax(node_value[mask]).item()
        index = mask.nonzero().squeeze()[selected_node]
        action = index.item()
        # take action
        next_ob, _, done = env.step(action)
        ob = next_ob


G = dc(nx_graph)
G.remove_nodes_from(torch.where(ob == 1)[0].cpu().numpy())
adj = nx.adjacency_matrix(G, dtype=float)
final_eigenvalue, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
final_loss = total_loss[-1] if total_loss else 0
final_neighbors = compute_total_degree(nx_graph, torch.where(ob == 1)[0].cpu().numpy())
print(f"Final eigenvalue: {final_eigenvalue}")
print(f"Final neighbors: {final_neighbors}")
print(f"Final loss: {final_loss}")


import matplotlib.pyplot as plt
import os

folder_path = 'plots/' + 'seed' + str(args.seed)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

version_name = args.reward + "lr_" + str(lr) + "_budget_" + str(budget) + "_max_episodes_" + str(max_episodes) + "_max_epsilon_" + str(max_epsilon) + "_min_epsilon_" + str(min_epsilon) + "_epsilon_decay_" + str(epsilon_decay) + "_batch_size_" + str(batch_size) + "_update_target_iters_" + str(update_target_iters) + "_buffer_size_" + str(args.buffer_size) + "_seed_" + str(args.seed)
eigen_vals_plot = folder_path + '/eigenvalues_plot_' + version_name + '.png'
total_loss_plot = folder_path + '/total_loss_plot_' + version_name + '.png'
neighbors_plot = folder_path + '/neighbors_plot_' + version_name + '.png'

# Plotting eigen_vals
plt.figure(figsize=(10, 5))
plt.plot(eigen_vals, label='Eigenvalues')
plt.xlabel('Index')
plt.ylabel('Eigenvalues')
plt.title('Plot of Eigenvalues')
plt.legend()
plt.grid(True)
plt.savefig(eigen_vals_plot)  # Save the plot as an image file
plt.close()

# Plotting total_loss
plt.figure(figsize=(10, 5))
plt.plot(total_loss, label='Total Loss', color='orange')
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Plot of Total Loss')
plt.legend()
plt.grid(True)
plt.savefig(total_loss_plot)  # Save the plot as an image file
plt.close()

# Plotting neighbors
plt.figure(figsize=(10, 5))
plt.plot(neighbors, label='Number of Neighbors', color='green')
plt.xlabel('Index')
plt.ylabel('Number of Neighbors')
plt.title('Plot of Number of Neighbors')
plt.legend()
plt.grid(True)
plt.savefig(neighbors_plot)  # Save the plot as an image file
plt.close()
