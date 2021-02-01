from dataset import Mydatasets
from layer import NodeCensNet, EdgeCensNet, CensNet
import numpy as np
import torch

nodedataset = np.arange(1, 11).reshape(5, 2)
nodedataset2 = np.arange(11, 21).reshape(5, 2)

nodedataset = np.stack([nodedataset, nodedataset2])

edgedataset = np.arange(1, 2 * 3*2+1).reshape(2*3, 2)
edgedataset2 = np.arange(2 * 3*2+1, 2 * 3*4+1).reshape(2*3, 2)

edgedataset = np.stack([edgedataset, edgedataset2])

# 有向グラフを想定
node_adj = np.array([[0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])

edge_adj = np.array([[0, 1, 1, 1, 0, 0],
                     [1, 0, 1, 0, 1, 0],
                     [1, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1],
                     [0, 0, 1, 1, 1, 0]])

# D+Iされている
D_v = np.array([[4, 0, 0, 0, 0],
                [0, 4, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]])

D_e = np.array([[4, 0, 0, 0, 0, 0],
                [0, 4, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 4]])

T = np.array([[1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1],
              [1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 1]])

datasets = Mydatasets(nodedataset, edgedataset,
                      node_adj, edge_adj, D_v, D_e, T)

trainloader = torch.utils.data.DataLoader(datasets, batch_size=2, shuffle=True)

nodenet = NodeCensNet(2, 3, 2)
edgenet = EdgeCensNet(2, 3, 3)
censenet = CensNet(2, 2, 3, 3)
for node, edge, node_adj, edge_adj, D_v, D_e, T in trainloader:
    """
    out = nodenet(node, edge, node_adj, D_v, T)
    out = edgenet(out, edge, edge_adj, D_e, T)
    print(out)
    """
    out = censenet(node, edge, node_adj, edge_adj, D_v, D_e, T)
    print(out)
