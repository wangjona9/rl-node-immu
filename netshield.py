import numpy as np
import networkx as nx
import torch 
from scipy.sparse.linalg import eigs
import pickle

def netshield(A, k):
    n = A.shape[0]
    
    # Eigenvalue and eigenvector
    lambda_1, u = eigs(A, k=1, which='LR') # Leading eigenvalue
    lambda_1 = np.real(lambda_1[0])
    u = np.real(u[:, 0])
    
    # Precompute score component
    v = np.zeros(n) 
    for j in range(n):
        v[j] = (2 * lambda_1 - A[j, j]) * (u[j] ** 2)
    
    S = []  # Selected node set
    
    for _ in range(k):
        b = np.zeros(n)
        if S:
            B = A[:, S]
            b = B @ u[S]
        
        scores = np.full(n, 0.0)
        for j in range(n):
            if j in S:
                scores[j] = -1.0
            else:
                scores[j] = v[j] - 2 * b[j] * u[j]
        
        i = np.argmax(scores)
        S.append(i)
    
    return S

