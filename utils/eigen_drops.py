import networkx as nx
import numpy as np
import scipy.sparse as sparse
from random import choice
import random
import copy
from torch.nn.functional import softmax
import torch
from torch_geometric.utils import subgraph, to_networkx
from copy import deepcopy as dc

class EigenDrop(torch.nn.Module):
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
        self.is_true = True

    def forward(self, graph, node_index, method, ob=None):
        
        if ob is not None:

            # G.remove_nodes_from(torch.where(ob[:,0] == 1)[0].cpu().numpy())
            mask = (node_index == 1).squeeze().cpu().numpy()
            adj = copy.deepcopy(self.adj)
            # import ipdb; ipdb.set_trace()
            adj[mask, :] = 0
            adj[:, mask] = 0
            
            eigenval, eigenvec = sparse.linalg.eigsh(adj, k=1, which='LA')
            

            if not self.is_true:
                mask = (self.old_index == 1).squeeze().cpu().numpy()
                adj = copy.deepcopy(self.adj)
                # import ipdb; ipdb.set_trace()
                adj[mask, :] = 0
                adj[:, mask] = 0
                old_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
                self.ori_eigenval = old_eigenval.item()

            eigen_drop = self.ori_eigenval - eigenval.item()

            if eigen_drop<0:
                eigen_drop = 0
            else:
                self.ori_eigenval = eigenval.item()

            self.eigenval = eigenval.item()
            self.eigenvec = torch.from_numpy(eigenvec).to(self.device).squeeze()
            self.A = torch.from_numpy(self.adj.todense()).to(self.device)
            
            self.old_index = node_index
            self.is_true = True
            return eigen_drop

        self.old_index = node_index
        self.is_true = False
        if method == 'uBCu':
            return NewEigenDropApprox(node_index, self.eigenval, self.eigenvec, self.adj, method=1)
        elif method == 'my':
            return NewEigenDropApprox(node_index, self.eigenval, self.eigenvec, self.adj, method=2)
        elif method == 'ns':
            return ShieldValue(node_index, self.eigenval, self.eigenvec, self.A)
        elif method == 'cen':
            return centrality(node_index, self.eigenval, self.eigenvec, self.A)
            

def NewEigenDropApprox(node_index, eigenval, eigenvec, adj, method=1):

    num_nodes = node_index.shape[0]
    num_deleted = (node_index==1).sum()
    num_remain = num_nodes-num_deleted
    idx = torch.argsort(node_index.squeeze()).cpu().numpy()

    A = copy.deepcopy(adj)
    A = A[idx, :]
    A = A[:, idx]
    E = copy.deepcopy(A)
    E[num_remain:, num_remain:] = 0
    B = A[:num_remain, num_remain:]
    C = A[num_remain:, :num_remain]
    u = eigenvec[idx, 0]
    u = u[:num_nodes-num_deleted]
    t1 = eigenval[0] - eigenval[1]
    t = eigenval[0]

    u = u / (u.T @ u)
    Cu = C @ u
    
    uB = u.T @ B
    uBCu = uB @ Cu
    if method == 1:
        return uBCu
    else:
        term1 = np.linalg.norm((B @ Cu ) - uBCu*u)

        term2 = 8*sparse.linalg.norm(E)**2 / t1**2 - np.linalg.norm(Cu/t)**2
        term2 = np.sqrt(term2)

        eigen_drop = (uBCu + term1*term2) / t 
        return eigen_drop



def ShieldValue(node_index, eigenval, eigenvec, adj):
    
    mask = (node_index == 1).squeeze()
    term1 = 2*eigenval*(mask*(eigenvec**2)).sum()
    masked_eigenvec = (eigenvec * mask).reshape(-1,1)
    term2 = (adj * (masked_eigenvec @ masked_eigenvec.T)).sum()
    eigendrop = term1 - term2
    return eigendrop.item()

def centrality(node_index, eigenval, eigenvec, adj):

    cen = ((adj @ eigenvec)**2 / eigenval).squeeze()

    eigendrop = (cen[node_index.squeeze().cpu().numpy()==1]).sum()
    return eigendrop

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

def compute_eigenval(nx_graph, node_idx):
    G = dc(nx_graph)
    G.remove_nodes_from(node_idx)
    # import pdb; pdb.set_trace()
    adj = nx.adjacency_matrix(G, dtype=float)
    after_eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
    return after_eigenval
