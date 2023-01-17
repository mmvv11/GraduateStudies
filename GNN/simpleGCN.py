## PyTorch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# import torch.nn.functional as F
# import torch.utils.data as data
# from torch_geometric.data import Data
# import torch.optim as optim
# import networkx as nx
# from torch_geometric.utils.convert import to_networkx

node_features = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor(
    [
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ]
    ]
)

print("Node features:\n", node_features)
print("\nAdjacency matrix:\n", adj_matrix)

## Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out) ##  y=xAT+b

    def forward(self, node_features, adj_matrix):
        """
        Inputs:
            node_features - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        print("Number of neighbors per node:", num_neighbours)
        node_features = self.projection(node_features) ## HW
        node_features = torch.bmm(adj_matrix, node_features) # Matrix multiplication between A (Adjacent Matrix) and HW (note features)
        node_features = node_features / num_neighbours
        return node_features


if __name__ == "__main__":
    layer = GCNLayer(c_in=2, c_out=2)
    layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]]) #Initialized the weigths to the identity matrix to let us validate the code.
    layer.projection.bias.data = torch.Tensor([0., 0.]) #bias term equal to zero

    with torch.no_grad():
        out_features = layer(node_features, adj_matrix)

    print("Adjacency matrix", adj_matrix)
    print("Input features", node_features)
    print("Output features", out_features)
