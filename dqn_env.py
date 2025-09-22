from utils.segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import torch
import networkx as nx
import random
from scipy import sparse
from copy import deepcopy as dc


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
        oracle_calls = 100,
        reward = 'shield'
        ):
        self.budget = budget
        self.oracle_calls = oracle_calls
        self.device = device
        self.nx_g = graph
        self.reward = reward
        
        
    def step(self, action):
        reward, done = self._take_action(action)    
        ob = self._build_ob()

        return ob, reward, done
    
    def _take_action(self, action):
        
        
        # compute reward and solution
        previous_deleted_nodes = (self.x == 1).long()
        reward = self._reward_compute(previous_deleted_nodes, action)
        self.x[action] = 1
        done = self._check_done()

        return reward, done

    def _reward_compute(self, node_index, action, episode=None):
        
        if self.reward == 'shield':
            if episode is not None:
                if episode % self.oracle_calls == 0: 
                    reward = EigenDrop(self.nx_g, node_index, action)
                    self.old_reward += reward
            else:
                
                tmp = dc(node_index)
                tmp[action] = 1
                new_reward = self.evaluation(tmp)
                reward = new_reward - self.old_reward
                self.old_reward = new_reward
        else:
            reward = SV(self.nx_g, node_index, action)

        return reward

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
        
        self.evaluation = ShieldValue(self.nx_g, self.device)
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