import networkx as nx
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, subgraph, degree
from torch_geometric.transforms import LargestConnectedComponents, AddRandomWalkPE
from torch_geometric.datasets import CitationFull, Amazon, Planetoid, TUDataset
from torch_geometric.data import Data
import torch.nn as nn
import torch
import os

class GraphLoader(nn.Module):
    def __init__(self, root: str, graph_name: str, args=None) -> None:
        # root: path for storing the data
        # graph_name: name of graph you wish to load
        # args: a list of parameters when loading random graphs

        super().__init__()
        self.root = root
        self.graph_name = graph_name
        self.args = args
    
    def forward(self):
        if self.graph_name in set(['Cora', 'Citeseer', 'DBLP', 'PubMed']):
            dataset = Planetoid(root='~/', name=self.graph_name)
            data = dataset[0]
        elif self.graph_name in set(['Computers', 'Photo']):
            dataset = Amazon(root=self.root, name=self.graph_name)
            data = dataset[0]
        elif self.graph_name == 'reddit':
            data = TUDataset(root=self.root, name='REDDIT-BINARY')
        elif self.graph_name == 'BA':
            data = BAGraph(self.args.num_nodes, self.args.num_edges)
        elif self.graph_name == 'ER':
            data = ERGraph(self.args.num_nodes, self.args.edge_prob)
        return data
    
def load_highschool():
    with open('datasets/out.moreno_highschool_highschool') as f:
        data = f.readlines()[2:]
        src = []
        dst = []
        for line in data:
            tmp = line.split(' ')
            src.append(int(tmp[0]) - 1)
            dst.append(int(tmp[1]) - 1)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def load_facebook():
    with open('datasets/fb-messages.edges') as f:
        data = f.readlines()
        src = []
        dst = []
        for line in data:
            tmp = line.split(' ')
            src.append(int(tmp[0]))
            dst.append(int(tmp[1]))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def load_infect():
    with open('datasets/infectious.edges') as f:
        data = f.readlines()
        src = []
        dst = []
        for line in data:
            tmp = line.split(' ')
            src.append(int(tmp[0]))
            dst.append(int(tmp[1]))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def load_graph(data_name, dim=128):
    if data_name == 'Cora':
        graph = GraphLoader(root='datasets/',graph_name="Cora")
        graph = graph()

        mask = graph.y==1
        graph.edge_index = subgraph(mask, graph.edge_index, relabel_nodes=True)[0]
        graph.x = graph.x[mask]
        graph.train_mask = graph.train_mask[mask]
        graph.val_mask = graph.val_mask[mask]
        graph.test_mask = graph.test_mask[mask]

        trans = LargestConnectedComponents()
        graph = trans(graph)
        random_walk_pe_cora = torch.load("datasets/random_walk_pe_cora.pt")
        node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes)
        nx_graph = to_networkx(graph)
        num_nodes = graph.x.shape[0]
        node_idx = torch.arange(num_nodes)
    else:
        if data_name == 'highschool':
            file_path = 'datasets/out.moreno_highschool_highschool'
            num_nodes = 70
        elif data_name == 'facebook':
            num_nodes = 1899
            file_path = 'datasets/fb-messages.edges'
        elif data_name == 'infect':
            num_nodes = 410
            file_path = 'datasets/infect-dublin.edges'
        else:
            raise ValueError('Invalid data name')


        with open(file_path) as f:
            data = f.readlines()
            src = []
            dst = []
            for line in data:
                if data_name == 'facebook':
                    tmp = line.split(',')
                else:
                    tmp = line.split(' ')
                src.append(int(tmp[0]) - 1)
                dst.append(int(tmp[1]) - 1)

            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
            node_idx = torch.arange(num_nodes)
            graph = Data(x=node_idx, edge_index=edge_index)
            trans = LargestConnectedComponents()
            graph = trans(graph)
            num_nodes = graph.x.shape[0]
            node_idx = torch.arange(num_nodes)
            nx_graph = to_networkx(graph)
            node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes)


            if os.path.exists(f"datasets/random_walk_pe_{data_name}_{dim}.pt"):
                random_walk_pe = torch.load(f"datasets/random_walk_pe_{data_name}_{dim}.pt")
            else:
                print(f"Generating random walk positional encoding for {data_name} graph")
                graph = AddRandomWalkPE(walk_length=dim)(graph)
                random_walk_pe = graph.random_walk_pe
                torch.save(random_walk_pe, f"datasets/random_walk_pe_{data_name}_{dim}.pt")
                print(f"Random walk positional encoding for {data_name} graph saved")

    return graph, nx_graph, random_walk_pe, node_degrees, num_nodes, node_idx