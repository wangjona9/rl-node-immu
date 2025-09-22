import argparse
import torch
import torch.nn as nn
from data.graph_loader import GraphLoader, load_highschool
from rl.env import NodeImmunization
from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl.gnn import RGCN, g2vec, MyModel
from tqdm import tqdm
import csv
import random
import math
import networkx as nx
import numpy as np
from copy import deepcopy as dc
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, subgraph, degree
from torch_geometric.transforms import LargestConnectedComponents, AddRandomWalkPE
import scipy.sparse as sparse
from torch.distributions import Categorical
from torch_geometric.loader import DataLoader

def polyak_avg(dict_1, dict_2, tau=0.999):
    for key in dict_1.keys():
        dict_1[key] = tau*dict_1[key] + (1-tau)*dict_2[key]
    return dict_1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-name", 
    default="Cora",
    help="directory to store validation and test datasets",
    type=str
    )
parser.add_argument(
    "--device",
    default=0,
    help="id of gpu device to use",
    type=int
    )
parser.add_argument(
    "--budget",
    help="number of nodes to delete",
    type=int
    )
parser.add_argument(
    "--method",
    help="number of nodes to delete",
    default="ns",
    type=str
    )
parser.add_argument(
    "--lr",
    default=5e-3,
    help="learning rate",
    type=float
    )
parser.add_argument(
    "--epsilon",
    default=0.9,
    type=float
    )
parser.add_argument(
    "--max_del",
    default=4,
    help="maximum deletion of single iteration",
    type=int
    )
parser.add_argument(
    "--o2",
    default=10,
    help="when to call oracle 2",
    type=int
    )
parser.add_argument(
    "--steps",
    default=5,
    type=int
    )


args = parser.parse_args()
device = torch.device(args.device)

    ###############################
    ##     Hyper-parameters      ##
    ###############################

discount_rate = 0.9
model_path = "saved_models/q_model_degree_lr_" + str(args.lr) + args.method + "_" + str(args.budget) + "_del_" + str(args.max_del) + "_o2_" + str(args.o2) + ".pth"
csv_path = "saved_models/results_degree_lr_" + str(args.lr) + "_" + args.method + "_del_" + str(args.max_del) + "_o2_" + str(args.o2) +".csv"

# network achitecture
num_layers = 3
input_dim = 64
output_dim = 1
hidden_dim = 128

# Optimize-related
update_target_iters = 1024
lr = args.lr
max_iters = args.budget
max_episodes = 5000
max_grad_norm = 1.0
epsilon = 0.9

batch_size = 32
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 1 / 300

# PER parameters
alpha = 0.2
beta = 0.6
n_steps = args.steps


    ###############################
    ##      Construct Model      ##
    ###############################

# Data

### Cora
# graph = GraphLoader(root='data/',graph_name=args.data_name)
# graph = graph()

# mask = graph.y==1
# graph.edge_index = subgraph(mask, graph.edge_index, relabel_nodes=True)[0]
# graph.x = graph.x[mask]
# graph.train_mask = graph.train_mask[mask]
# graph.val_mask = graph.val_mask[mask]
# graph.test_mask = graph.test_mask[mask]

# trans = LargestConnectedComponents()
# graph = trans(graph)

# random_walk_pe_cora = torch.load("datasets/random_walk_pe_cora.pt").to(device)


# nx_graph = to_networkx(graph)
# graph = graph.to(device)
# num_nodes = graph.x.shape[0]
# node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes)
# node_idx = torch.arange(num_nodes).to(device)

### Highschool
from torch_geometric.data import Data
edge_index = load_highschool()
num_nodes = 70
edge_index = to_undirected(edge_index, num_nodes=num_nodes)
node_idx = torch.arange(num_nodes).to(device)
graph = Data(x=node_idx.cpu(), edge_index=edge_index)
nx_graph = to_networkx(graph)
node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes)
random_walk_pe = torch.load("datasets/random_walk_pe_highschool_128.pt").to(device)

# Enviroment
env = NodeImmunization(
    budget=args.budget,
    max_single_del=args.max_del,
    oracle_2=args.o2,
    max_epi_t=args.budget,
    device=device,
    method=args.method,
    graph=nx_graph,
    )

# Replay Buffer
buffer = PrioritizedReplayBuffer(num_nodes, batch_size=batch_size, alpha=alpha)

# Model 

# Structure2Vec
# q_value = g2vec(num_nodes=num_nodes,
#     hidden_dim=hidden_dim).to(device)
# target_q_value = g2vec(num_nodes=num_nodes,
#     hidden_dim=hidden_dim).to(device)

# embeddings + GCN
q_value = MyModel(num_nodes=num_nodes,
    hidden_dim=hidden_dim).to(device)
target_q_value = MyModel(num_nodes=num_nodes,
    hidden_dim=hidden_dim).to(device)

# RGCN
# q_value = RGCN(
#     num_relations=2,
#     input_dim=num_nodes,
#     hidden_dim=hidden_dim,
#     ).to(device)
# target_q_value = RGCN(
#     num_relations=2,
#     input_dim=num_nodes,
#     hidden_dim=hidden_dim,
#     ).to(device)

# initailize the embeddings and models
q_value.embeddings.weight.data = random_walk_pe
target_q_value.load_state_dict(q_value.state_dict())

# Optimizer
optimizer = torch.optim.Adam(q_value.parameters(), lr=lr)


with open(csv_path, "w") as csvfile:
    fields = ['episode', 'avg loss', 'eigenval']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    iters = 0
    verbose = False

    # episode start
    for episode in range(1, max_episodes+1):
        ob = env.register(graph)
        done = False
        mean_loss = []
        while not done:
            # Interaction 
            q_value.eval()
            with torch.no_grad():
                mask = (env.x == 0).squeeze()
                
                node_value = q_value(node_idx, graph.edge_index, ob)
                
                # choose action
                if epsilon > np.random.random():
                    # if 0.5 > np.random.random():
                        randomly_selected_node = random.choice(range(mask.sum().item()))
                        index = mask.nonzero().squeeze()[randomly_selected_node]
                        action = index

                    # The degree policy
                    # else:
                    #     indices = mask.nonzero().squeeze()
                    #     selected_node = torch.argmax(node_degrees[indices]).item()
                    #     action = indices[selected_node]
                else:
                    
                    selected_node = torch.argmax(node_value[mask]).item()
                    index = mask.nonzero().squeeze()[selected_node]
                    action = index
                    

                # take action
                next_ob, reward, done = env.step(action)
                
                # store the trajectory
                buffer.store(
                    ob.squeeze().cpu().numpy(),
                    action.item(),
                    reward,
                    next_ob.squeeze().cpu().numpy(),
                    done.item()
                )
                ob = next_ob

            count = 0
            avg_loss = 0
            for iters in range(4):
                # only start to train when we have enough samples
                if buffer.size - 8 > buffer.batch_size:

                    ########## data preparation ###############################################
                    verbose = True
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
                        G.done = torch.tensor([batch_data['done'][i]], dtype=torch.bool)
                        G.indices = batch_data["indices"][i]
                        G.weights = batch_data["weights"][i]
                
                        batch_dataset.append(G)

                    # use multiple steps
                    if n_steps > 1:
                        n_steps_batch_dataset = []
                        n_steps_data = buffer.sample_batch_from_idxs(batch_data["indices"])
                        for i in range(batch_data["obs"].shape[0]):
                            
                            G = dc(graph).cpu()
                            G.x = node_idx.cpu()
                            next_obs = torch.from_numpy(n_steps_data['next_obs'][i])
                            obs = torch.from_numpy(n_steps_data['obs'][i])
                            G.obs = obs
                            G.next_obs = next_obs
                            G.rews = torch.tensor([n_steps_data['rews'][i]])
                            G.y = torch.zeros(num_nodes, 1).squeeze()
                            G.y[int(n_steps_data['acts'][i])] = 1
                            G.done = torch.tensor([n_steps_data['done'][i]], dtype=torch.bool)
                    
                            n_steps_batch_dataset.append(G)

                    #########################################################################
                    


                    ################## Q-learning #############################################
                    loader1 = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)
                    for batch in loader1:
                        # Target
                        batch = batch.to(device)
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

                        # Train
                        predicted_value = q_value(batch.x, batch.edge_index, batch.obs) 
                        elementwise_loss = 0.5*((target - predicted_value[batch.y==1].squeeze())**2)

                    # use multiple steps
                    if n_steps > 1:
                        loader2 = DataLoader(n_steps_batch_dataset, batch_size=batch_size, shuffle=False)
                        for batch2 in loader2:
                            # Target
                            batch2 = batch2.to(device)
                            with torch.no_grad():
                                target = target_q_value(batch2.x, batch2.edge_index, batch2.next_obs).detach()
                                predicted_value_next = q_value(batch2.x, batch2.edge_index, batch2.next_obs).detach()
                                
                                actions = []
                                for i in range(len(batch2.rews)):
                                    mask = (batch2.batch == i).squeeze()
                                    feasible_actions = (batch2.obs[mask] == 0).squeeze()
                                    selected_node = torch.argmax(predicted_value_next[mask][feasible_actions]).item()
                                    index = feasible_actions.nonzero().squeeze()[selected_node].item()
                                    index = mask.nonzero().squeeze()[index].item()
                                    actions.append(index)   # choose a feasible action

                                discount_rate_n = discount_rate ** n_steps
                                discounted_rewards = discount_rate_n * target[actions].squeeze()
                                discounted_rewards[batch2.done] = 0     # set to 0 if it it the last step
                                target = batch2.rews + discounted_rewards

                            # Train
                            predicted_value = q_value(batch2.x, batch2.edge_index, batch2.obs) 
                            elementwise_n_loss = 0.5*((target - predicted_value[batch2.y==1].squeeze())**2)
                            elementwise_loss += elementwise_n_loss

                    
                    loss = (batch.weights * elementwise_loss).mean()
                    avg_loss += loss.detach().item()
                    count += 1

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                            q_value.parameters(), 
                            max_grad_norm
                            )
                    optimizer.step()
                    ######################################################################################



                    # Update priorities
                    loss_for_prior = elementwise_loss.detach().cpu().numpy()
                    new_priorities = loss_for_prior + 1e-6
                    buffer.update_priorities(batch.indices.cpu().numpy(), new_priorities)

                # logging loss
                mean_loss.append(avg_loss/(count+1e-8))

                # Set two networks equal for every "update_target_iters" steps
                iters += 1
                if iters % update_target_iters == 0:
                    target_q_value.load_state_dict(q_value.state_dict())
                    iters = 0

                # Polyak averaging
                # target_q_value.load_state_dict(polyak_avg(target_q_value.state_dict(), q_value.state_dict()))

                # Update epsilon
                epsilon = max(
                    min_epsilon, epsilon - (
                        max_epsilon - min_epsilon
                    ) * epsilon_decay
                )


        # Increase Beta
        fraction = min(episode / max_episodes, 1.0)
        beta = beta + fraction * (1.0 - beta)

        # logging the true eigen drop
        G = dc(nx_graph)
        G.remove_nodes_from(torch.where(ob == 1)[0].cpu().numpy())
        adj = nx.adjacency_matrix(G, dtype=float)
        eigenval, _ = sparse.linalg.eigsh(adj, k=1, which='LA')
        
        if verbose:
            torch.save(q_value.state_dict(), model_path)
            writer.writerow({
                'episode': episode, 
                'avg loss': np.mean(mean_loss),
                'eigenval': eigenval.item()})
            print("Episode: {}/{} , Avg Loss: {:.4f}, Eigenval: {:.4f}, Mean: {:.4f}, STD: {:.4f}".format(
            episode, max_episodes, 
            np.mean(mean_loss), 
            eigenval.item(), 
            predicted_value.detach().mean().item(),
            predicted_value.detach().std().item()
            )
            )
