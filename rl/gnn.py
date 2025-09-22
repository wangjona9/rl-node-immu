import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv, GATConv, GCNConv


def get_masked_adj(edge_index, node_type):
        src, dst = edge_index[0], edge_index[1]
        src_mask = node_type[src] == 0
        dst_mask = node_type[dst] == 0
        mask = src_mask * dst_mask
        
        return edge_index[:, mask.squeeze()]


class g2vec(nn.Module):
    def __init__(self, num_nodes,
        hidden_dim,
        output_dim=1,
        num_layers=3,
        dropout=0.5):
        super(g2vec, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(hidden_dim, hidden_dim)
            # FastRGCNConv(input_dim, hidden_dim, num_relations=num_relations)
            )
        # self.layers.append(
        #      GATConv(hidden_dim, hidden_dim)
        #     # FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        #     )

        self.layers.append(
            GATConv(hidden_dim, output_dim)
            # FastRGCNConv(hidden_dim, output_dim, num_relations=num_relations)
            )
        

            
    def forward(self, h, edge_index, node_type):
        adj = get_masked_adj(edge_index, node_type)

        h = self.embeddings(h)
        h = self.layers[0](h, adj)
        h = F.relu(h)
        h = self.layers[1](h, adj)
        # h = F.relu(h)
        # h = self.layers[2](h, adj)

        # h = self.layers[0](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.layers[1](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.layers[2](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.bn(h)
        # h = self.layers[3](h)
            
        return h
    
class MyModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim,
        output_dim=1,
        num_layers=3,
        dropout=0.5
        ):
        super(MyModel, self).__init__()
        self.GNNs = nn.ModuleList()
        self.embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.GNNs.append(
            GCNConv(hidden_dim, 4*hidden_dim)
            )
        self.GNNs.append(
            GCNConv(4*hidden_dim, 4*hidden_dim)
            )
        self.GNNs.append(
            GCNConv(4*hidden_dim, 3*hidden_dim)
            )

        self.prediction_head = nn.Sequential(
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(2*hidden_dim),
            nn.Linear(2*hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, node_type):
        adj = get_masked_adj(edge_index, node_type)
        mask = (node_type == 0).squeeze()

        z = self.embeddings(x)
        z = self.GNNs[0](z, adj)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.GNNs[1](z, adj)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.GNNs[2](z, adj)
        z = F.relu(z)

        
        z = z[mask]
        # add_pooling = z.sum(dim=0, keepdim=True).repeat(mask.shape[0], 1)
        mean_pooling = z.mean(dim=0, keepdim=True).repeat(mask.shape[0], 1)
        # max_pooling = z.max(dim=0, keepdim=True).values.repeat(mask.shape[0], 1)
        
        z = torch.cat((mean_pooling, self.embeddings(x)), dim=1)
        z = self.prediction_head(z)
        return z

    def reset_noise(self):
        for module in self.children():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()



class RGCN(nn.Module):
    def __init__(
        self,
        num_relations,
        input_dim,
        hidden_dim,
        output_dim=1,
        num_layers=3,
        dropout=0.5
        ):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.embeddings = nn.Embedding(input_dim, hidden_dim)
        self.layers.append(
            FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            )
        # self.layers.append(
        #     FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        #     )

        self.layers.append(
            FastRGCNConv(hidden_dim, output_dim, num_relations=num_relations)
            )
        # self.layers.append(
        #     nn.Linear(hidden_dim, hidden_dim)
        #     )
        # self.layers.append(
        #     nn.Linear(hidden_dim, output_dim)
        #     )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def get_edge_type(self, edge_index, node_type):
        src, dst = edge_index[0], edge_index[1]
        return (node_type[src] == node_type[dst]).squeeze().long()
    
    def get_edge_type_4(self, edge_index, node_type):
        src, dst = edge_index[0], edge_index[1]

        # Determine the edge type based on the node types
        edge_type = 2 * node_type[src] + node_type[dst]
        
        return edge_type.squeeze().long()

            
    def forward(self, x, edge_index, node_type):
        edge_type = self.get_edge_type(edge_index, node_type)
        
        x = self.embeddings(x)
        x = self.layers[0](x, edge_index, edge_type)
        x = F.relu(x)
        x = self.layers[1](x, edge_index, edge_type)
        # h = F.relu(h)
        # h = self.layers[2](h, edge_index)

        # h = self.layers[0](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.layers[1](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.layers[2](h, edge_index, edge_attr=edge_type)
        # h = F.relu(h)
        # h = self.bn(h)
        # h = self.layers[3](h)
            
        return x
    