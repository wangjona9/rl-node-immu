import networkx as nx
import numpy as np
import scipy.sparse as sparse
import scipy as sp
import random
import copy


def randomPick(G: nx.graph, k: int) -> list[int]:
    return random.sample(list(G.nodes), k)

def degree(G: nx.graph, k: int) -> list[int]:
    degree_list  = sorted(dict(G.degree).items(), key=lambda x:-x[1])[:k]
    return [node[0] for node in degree_list]

def netShield(graph: nx.graph, k: int) -> list[int]:
    """Return the nodes that have the largest 'shield value'.

    References
    ----------

    [1] Chen Chen, Hanghang Tong, B. Aditya Prakash, Charalampos
    E. Tsourakakis, Tina Eliassi-Rad, Christos Faloutsos, Duen Horng Chau:
    Node Immunization on Large Graphs: Theory and Algorithms. IEEE
    Trans. Knowl. Data Eng. 28(1): 113-126 (2016)

    Notes
    -----

    This version only works on graphs with no self-loops.

    """
    # Matrix computations require node labels to be consecutive integers,
    # so we need to (i) convert them if they are not, and (ii) preserve the
    # original labels as an attribute.
    graph = nx.convert_node_labels_to_integers(
        graph, label_attribute='original_label')

    # The variable names here match the notation used in [1].
    A = nx.to_scipy_sparse_array(graph).astype('d')
    nodes = [n for n in graph]
    lambda_, u = sparse.linalg.eigs(A, k=1)
    lambda_ = lambda_.real
    u = abs(u.real)
    v = 2 * lambda_ * u**2

    # The first node is just the one with highest eigenvector centrailty
    first_idx = np.argmax(v)
    S = [nodes[first_idx]]
    score = np.zeros(graph.order())
    score[first_idx] = -1
    for _ in range(k - 1):
        # Update score
        B = A[:, S]
        b = B.dot(u[S])
        for idx, j in enumerate(nodes):
            if j in S:
                score[idx] = -1
            else:
                score[idx] = (v[idx] - 2 * b[idx] * u[idx])[0]
        # The remaining nodes are the ones that maximize the score
        S.append(nodes[np.argmax(score)])

    # Return the original labels
    return [graph.nodes[s]['original_label'] for s in S]

def centrality(G: nx.graph, k: int) -> list[int]:
    adj = nx.adjacency_matrix(G, dtype=float)
    eigenval, eigenvec = sparse.linalg.eigsh(adj, k=1, which='LA')
    cen = ((adj @ eigenvec)**2 / eigenval).squeeze()
    targets = np.argsort(-cen)[:k]
    return [list(G.nodes)[node] for node in targets]

def myval1(G: nx.graph, k: int) -> list[int]:
    adj = nx.adjacency_matrix(G, dtype=float)
    eigenval, eigenvec = sparse.linalg.eigsh(adj, k=2, which='LA')
    eigengap = eigenval[0] - eigenval[1]
    eigenval = eigenval[0]
    eigenvec = eigenvec[:, 0]

    Cu = adj @ eigenvec
    uBCu = Cu**2
    term1 = np.sum((Cu*adj - uBCu*eigenvec)**2, axis=-1)
    term1 = np.sqrt(term1)
    term2 = 8*np.sqrt(np.sum(adj**2, 1)) / eigengap**2 - (Cu/eigenval)**2
    term2 = np.sqrt(term2)
    val = ((uBCu + term1*term2) / eigenval).squeeze()
    return delete_indSet(G, val, k)

def myval2(G: nx.graph, k: int) -> list[int]:
    adj = nx.adjacency_matrix(G, dtype=float)
    eigenval, eigenvec = sparse.linalg.eigsh(adj, k=1, which='LA')
    # eigengap = eigenval[0] - eigenval[1]
    # eigenval = eigenval[0]
    # eigenvec = eigenvec[:, 0]

    Cu = adj @ eigenvec
    uBCu = (Cu**2/ eigenval).squeeze()
    # term1 = np.sum((Cu*adj - uBCu*eigenvec)**2, axis=-1)
    # term1 = np.sqrt(term1)
    # term2 = 8*np.sqrt(np.sum(adj**2, 1)) / eigengap**2 - (Cu/eigenval)**2
    # term2 = np.sqrt(term2)
    # val = (uBCu - term1*term2) / eigenval 
    return delete_indSet(G, uBCu, k)

def delete_indSet(G: nx.graph, val: list[int], k: int) -> list[int]:
    idx = np.argsort(-val).tolist()
    indSet = []
    for i in range(k): 
        indSet.append(idx.pop(0))
        # import ipdb; ipdb.set_trace()
        for node in list(G.neighbors(list(G.nodes)[indSet[-1]])):
            if node in idx:
                idx.remove(node)
    return [list(G.nodes)[node] for node in indSet]


class topk:
    def __init__(self, G: nx.graph, nodes_to_delete: int, k: int, method: str ='random') -> None:
        if method == 'random':
            self.metric = randomPick
        elif method == 'degree':
            self.metric = degree
        elif method == 'ns':
            self.metric = netShield
        elif method == 'centrality':
            self.metric = centrality
        elif method == 'my1':
            self.metric = myval1
        elif method == 'uBCu':
            self.metric = myval2

        self.G = G
        self.nodes_to_delete = nodes_to_delete
        self.k = k
        adj = nx.adjacency_matrix(G, dtype=float)
        self.eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')

        self.plan = [k]*(nodes_to_delete//k)
        if nodes_to_delete%k != 0:
            self.plan.append(nodes_to_delete%k)
    
    def re_init(self):
        k = self.k
        nodes_to_delete = self.nodes_to_delete
        self.plan = [k]*(nodes_to_delete//k)
        if nodes_to_delete%k != 0:
            self.plan.append(nodes_to_delete%k)
    
    def go(self) -> None:
        G = copy.deepcopy(self.G)
        eigenvals = []
        idxs = []
        while len(self.plan) != 0:
            k = self.plan.pop(0)
            idx = self.metric(G, k)
            # import ipdb; ipdb.set_trace()
            G.remove_nodes_from(idx)
            adj = nx.adjacency_matrix(G, dtype=float)
            eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
            eigenvals.append(eigenval)
            idxs.append(idx)
        return eigenvals, idxs

        
        
            

        