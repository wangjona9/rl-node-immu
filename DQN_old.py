import torch
from torch_geometric.utils import to_networkx, to_undirected, degree
from torch_geometric.data import Data
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple
import networkx as nx
import scipy.sparse as sparse
from copy import deepcopy as dc
from torch_geometric.loader import DataLoader
import argparse

from rl.gnn import MyModel, RGCN
from data.graph_loader import GraphLoader, load_highschool
from utils.segment_tree import MinSegmentTree, SumSegmentTree

def parse_args():
    parser = argparse.ArgumentParser(description='Node Immunization with RL')
    parser.add_argument('--device', type=str, default='0', help='Device to use for computation')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--budget', type=int, default=5, help='Budget for node immunization')
    parser.add_argument('--max_episodes', type=int, default=5000, help='Maximum number of episodes')
    parser.add_argument('--max_epsilon', type=float, default=1.0, help='Starting value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--epsilon_decay', type=float, default=1/3000, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter for prioritized replay buffer')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for prioritized replay buffer')
    parser.add_argument('--update_target_iters', type=int, default=500, help='Interval for updating target network')
    parser.add_argument('--buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--oracle_calls', type=int, default=20, help='Number of oracle calls')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

args = parse_args()

# Set the random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

############################ DATA LOADING #########################
device = torch.device('cuda:'+args.device)
edge_index = load_highschool()
num_nodes = 70
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes).to(device)
graph = Data(x=node_idx.cpu(), edge_index=edge_index)
nx_graph = to_networkx(graph)
node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes)
random_walk_pe = torch.load("datasets/random_walk_pe_highschool_128.pt").to(device)

############################ MODEL #########################
hidden_dim = args.hidden_dim
lr = args.lr
q_value = MyModel(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
q_value.embeddings.weight.data = random_walk_pe
target_q_value = MyModel(num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
target_q_value.load_state_dict(q_value.state_dict())
optimizer = torch.optim.AdamW(q_value.parameters(), lr=lr)

############################ REPLAY BUFFER #########################
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 1.0
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int_)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.full(size, False, dtype=bool)
        
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        eps: float = 1e-6
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.eps = eps
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

############################ ENVIRONMENT #########################
class NodeImmunization(object):
    def __init__(
        self, 
        budget,
        device,
        graph,
        oracle_calls = 20,
        ):
        self.budget = budget
        self.oracle_calls = oracle_calls
        self.device = device
        self.nx_g = graph
        
        
    def step(self, action, episode=None):
        reward, done = self._take_action(action, episode)    
        ob = self._build_ob()

        return ob, reward, done
    
    def _take_action(self, action, episode=None):
        
        
        # compute reward and solution
        previous_deleted_nodes = (self.x == 1).long()
        reward = self._reward_compute(previous_deleted_nodes, action, episode)
        self.x[action] = 1
        done = self._check_done()

        return reward, done

    def _reward_compute(self, node_index, action, episode=None):
        
        # node_index = node_index.squeeze().cpu().numpy()
        # node_index = node_index.nonzero()[0]
        
        # mask = torch.where(node_index)[0].cpu().numpy()
        # mask = mask.tolist()
        # mask.append(action)
        
        # eigen_drop = compute_total_degree(self.nx_g, mask)
        
        # new_reward = eigen_drop - self.old_reward
        # if new_reward< 0:
        #     new_reward = 0

        # self.old_reward = eigen_drop

        if episode is not None and episode % self.oracle_calls == 0:
            new_reward = EigenDrop(self.nx_g, node_index, action)
        else:
            new_reward = SV(self.nx_g, node_index, action)

        
        return new_reward

    def _check_done(self):
        num_deleted = (self.x == 1).float()
        return num_deleted.sum() == self.budget
            
    def _build_ob(self):
        return self.x.float()
        
    def register(self, g, num_samples = 1):
        self.g = g
        self.num_samples = num_samples
        self.g.to(self.device)
        self.old_reward = 0
        
        num_nodes = self.g.x.shape[0]
        self.x = torch.zeros(
            num_nodes, 
            num_samples, 
            dtype = torch.long, 
            device = self.device
            )
        
        # self.evaluation = ShieldValue(self.nx_g, self.device)
        ob = self._build_ob()      
        return ob

def compute_total_degree(graph, nodes):
    unique_neighbors = set()
    for node in nodes:
        unique_neighbors.update(graph.neighbors(node))
    
    # Remove the nodes of interest from the unique neighbors set
    unique_neighbors.update(nodes)
    return len(unique_neighbors)

class ShieldValue(torch.nn.Module):
    def __init__(self, graph, device) -> None:
        super().__init__()
        self.device = device
        self.G = graph
        self.adj = nx.adjacency_matrix(self.G, dtype=float).tolil()
        eigenval, eigenvec = sparse.linalg.eigsh(self.adj, k=1, which='LA')
        self.eigenval = eigenval.item()
        self.ori_eigenval = eigenval.item()

        self.A = torch.from_numpy(self.adj.todense()).to(device)
        self.eigenvec = torch.from_numpy(eigenvec).to(device).squeeze()


    def forward(self, node_index):
        eigenval = self.eigenval
        eigenvec = self.eigenvec
        adj = self.A

        mask = (node_index == 1).squeeze()
        term1 = 2*eigenval*(mask*(eigenvec**2)).sum()
        masked_eigenvec = (eigenvec * mask).reshape(-1,1)
        term2 = (adj * (masked_eigenvec @ masked_eigenvec.T)).sum()
        eigendrop = term1 - term2
        return eigendrop.item()
    
def EigenDrop(graph, node_index, action):
    adj = nx.adjacency_matrix(graph, dtype=float).tolil()
    mask = (node_index == 1).squeeze().cpu().numpy()
    adj[mask, :] = 0
    adj[:, mask] = 0
    eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')

    adj[action, :] = 0
    adj[:, action] = 0
    eigenval_new, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    return eigenval.item() - eigenval_new.item()
    
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

    term1 = 2*eigenval*(mask*(eigenvec**2)).sum()
    masked_eigenvec = (eigenvec * mask).reshape(-1,1)
    term2 = (adj * (masked_eigenvec @ masked_eigenvec.T)).sum()
    eigendrop = term1 - term2
    return eigendrop.item()

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
budget = args.budget
verbose = args.verbose
beta = args.beta
alpha = args.alpha
oracle_calls = args.oracle_calls

shield_values = torch.load("datasets/shield_values_highschool.pt").to(device)


env = NodeImmunization(budget=budget, device=device, graph=nx_graph, oracle_calls=oracle_calls)
buffer = ReplayBuffer(num_nodes, size=args.buffer_size, batch_size=batch_size, gamma=discount_rate)
# buffer = PrioritizedReplayBuffer(num_nodes, size=args.buffer_size, batch_size=batch_size, alpha=alpha)

reward_history = []
step_count = 0

eigen_vals = []
total_loss = []
neighbors = []

for episode in range(max_episodes):
    episode_reward = 0
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
            with torch.no_grad():
                q_values = q_value(node_idx, graph.edge_index, ob)
                selected_node = torch.argmax(q_values[mask]).item()
                index = mask.nonzero().squeeze()[selected_node]
                action = index.item()

        next_ob, reward, done = env.step(action, episode*budget + t + 1)
        # import pdb; pdb.set_trace()
        buffer.store(ob.cpu().squeeze().numpy(), action, reward, next_ob.cpu().squeeze().numpy(), done)

        episode_reward += reward
        ob = next_ob

        if done:
            break
    
    num_neighbors = compute_total_degree(nx_graph, torch.where(ob == 1)[0].cpu().numpy())
    reward_history.append(episode_reward)

    if len(buffer) >= 2*batch_size:
        batch_data = buffer.sample_batch()
        # batch_data = buffer.sample_batch(beta)
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
            # G.weights = batch_data["weights"][i]
    
            batch_dataset.append(G)

        loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            # Target

            with torch.no_grad():
                
                target = target_q_value(batch.x, batch.edge_index, batch.next_obs).detach()
                predicted_value_next = q_value(batch.x, batch.edge_index, batch.next_obs).detach()
                
                actions = []
                for i in range(len(batch.rews)):
                    mask = (batch.batch == i).squeeze()
                    feasible_actions = (batch.obs[mask] == 0).squeeze()
                    selected_node = torch.argmax(predicted_value_next[mask][feasible_actions]).item()
                    index = feasible_actions.nonzero().squeeze()[selected_node].item()
                    index = mask.nonzero().squeeze()[index].item()
                    actions.append(index)   # choose a feasible action

                discounted_rewards = discount_rate*target[actions].squeeze()
                discounted_rewards[batch.done] = 0      # set to 0 if it it the last step
                target = batch.rews + discounted_rewards


        q_values = q_value(batch.x, batch.edge_index, batch.obs)
        q_values = q_values[batch.y==1].squeeze()

        elementwise_loss = torch.nn.functional.smooth_l1_loss(q_values, target, reduction='none')
        loss = torch.mean(elementwise_loss)
        # loss = torch.mean(elementwise_loss * batch.weights)
        # buffer.update_priorities(batch.indices.tolist(), elementwise_loss.detach().cpu().numpy() + buffer.eps)

        # if loss.item()>10:
        #     import pdb; pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_value.parameters(), max_grad_norm)
        optimizer.step()

        if verbose:
            G = dc(nx_graph)
            G.remove_nodes_from(torch.where(ob == 1)[0].cpu().numpy())
            adj = nx.adjacency_matrix(G, dtype=float)
            eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
            eigen_vals.append(eigenval.item())
            total_loss.append(elementwise_loss.mean().item())
            neighbors.append(num_neighbors)
            

            print("Episode: {}/{} , Avg Loss: {:.4f}, Eigenval: {:.4f}, Neighbor: {:.4f}, Mean: {:.4f}, STD: {:.4f}".format(
            episode, max_episodes, 
            total_loss[-1], 
            eigen_vals[-1], 
            neighbors[-1],
            predicted_value_next.detach().mean().item(),
            predicted_value_next.detach().std().item()
            )
            )

    if episode % update_target_iters == 0:
        target_q_value.load_state_dict(q_value.state_dict())
    
    
    
    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    # fraction = min(episode / max_episodes, 1.0)
    # beta = beta + fraction * (1.0 - beta)

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

print("Eigenvalues: {:.4f} ± {:.4f}".format(np.mean(eigen_vals[-500:]), np.std(eigen_vals[-500:])))



import matplotlib.pyplot as plt
import os

folder_path = 'plots/' + 'seed' + str(args.seed)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

version_name = "old_lr_" + str(lr) + "_budget_" + str(budget) + '_oracle_' + str(oracle_calls) + "_max_episodes_" + str(max_episodes) + "_max_epsilon_" + str(max_epsilon) + "_min_epsilon_" + str(min_epsilon) + "_epsilon_decay_" + str(epsilon_decay) + "_batch_size_" + str(batch_size) + "_update_target_iters_" + str(update_target_iters) + "_buffer_size_" + str(args.buffer_size) + "_seed_" + str(args.seed)
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
