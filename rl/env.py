import torch
from torch_geometric.nn import SimpleConv

from utils.eigen_drops import EigenDrop
from collections import namedtuple
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader


def adjust_action(action, action_prob, constant):
    true_indices = torch.nonzero(action).squeeze(1)
    indices = action_prob[true_indices].argsort(descending=True)[constant:]
    action[true_indices[indices]] = 0
    return action
    
class NodeImmunization(object):
    def __init__(
        self, 
        budget,
        max_single_del,
        oracle_2,
        max_epi_t, 
        device,
        method,
        graph,
        ):
        self.budget = budget
        self.oracle_2 = oracle_2
        self.max_single_del = max_single_del
        self.max_epi_t = max_epi_t
        # self.max_num_nodes = max_num_nodes
        self.device = device
        self.method = method
        self.nx_g = graph
        
        
    def step(self, action):
        reward, done = self._take_action(action)    
        ob = self._build_ob()

        return ob, reward, done
    
    def _take_action(self, action):
        self.x[action] = 1
        self.t += 1
        
        done = self._check_done()
        self.t += 1
        self.step_t += 1

        # compute reward and solution
        deleted_nodes = (self.x == 1).float()
        reward = self._reward_compute(deleted_nodes)

        return reward, done

    def _reward_compute(self, node_index):
        
        if self.step_t % self.oracle_2 == 0:
            eigen_drop = self.evaluation(self.g, node_index, self.method, ob=self.x)
            new_reward = eigen_drop
        else:
            eigen_drop = self.evaluation(self.g, node_index, self.method)
            new_reward = eigen_drop - self.old_reward
            if new_reward< 0:
                new_reward = 0
            # print(self.old_reward, eigen_drop, new_reward)
            self.old_reward = eigen_drop
            
        return new_reward

    def _check_done(self):
        num_deleted = (self.x == 1).float()
        return num_deleted.sum() == self.budget
            

                
    def _build_ob(self):
        return self.x.float()
        
    def register(self, g, num_samples = 1, batch_size = 1):
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
        self.t = torch.zeros(
            num_nodes, 
            num_samples, 
            dtype = torch.long, 
            device = self.device
            )
        
        self.evaluation = EigenDrop(self.nx_g, self.device)
        ob = self._build_ob()
        self.step_t = 0
            
        return ob