import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from GCN.layer import CensNet
from tools.graph import make_T_matrix, make_edge_adj, make_D_matrix
from collections import namedtuple
import torch.distributions as tdist
from .condition import easy_dev
from env.gym_barfem import BarFemGym


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


def adopt_batch_norm(node, edge, b1, b2):
    node = node.permute(0, 2, 1)  # BatchNormを適用するための手順
    edge = edge.permute(0, 2, 1)  # BatchNormを適用するための手順
    node = b1(node)
    edge = b2(edge)
    node = node.permute(0, 2, 1)
    edge = edge.permute(0, 2, 1)

    return node, edge


class Select_node1_model(nn.Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(Select_node1_model, self).__init__()
        self.GCN1 = CensNet(node_in_features, edge_in_features, node_out_features,
                            edge_out_features)
        self.GCN2 = CensNet(node_out_features, edge_out_features,
                            node_out_features, edge_out_features)
        self.predict_v1 = torch.nn.Linear(node_out_features, node_out_features)
        self.predict_v2 = torch.nn.Linear(node_out_features, 1)

        #self.b1 = torch.nn.BatchNorm1d(node_out_features)
        #self.b2 = torch.nn.BatchNorm1d(edge_out_features)
        #self.b3 = torch.nn.BatchNorm1d(node_out_features)
        #self.b4 = torch.nn.BatchNorm1d(edge_out_features)

        self.saved_actions = []
        self.rewards = []

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T, remove_index=False):
        """
        forward of both actor and critic
        """
        node, edge = self.GCN1(node, edge, node_adj, edge_adj, D_v, D_e, T)
        #node, edge = adopt_batch_norm(node, edge, self.b1, self.b2)
        node, edge = self.GCN2(node, edge, node_adj, edge_adj, D_v, D_e, T)
        #node, edge = adopt_batch_norm(node, edge, self.b3, self.b4)
        emb_node = node.clone()
        node = F.relu(self.predict_v1(node))  # 1*node_num*node_out_features
        node = self.predict_v2(node)  # 1*node_num
        node = node.reshape((-1, node.size(1)))
        if remove_index is not False:
            node = torch.cat(
                [node[:, 0:remove_index], node[:, remove_index + 1:]], 1)
        node = F.softmax(node, dim=-1)

        return emb_node, node


class Select_node2_model(torch.nn.Module):
    def __init__(self, node_in_features, emb_size):
        super(Select_node2_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, 1)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_node):
        x = F.relu(self.layer1(emb_node))  # 1*node_num*emb_size
        x = self.layer2(x)  # 1*node_num*1
        x = x.reshape((1, -1))
        x = F.softmax(x, dim=-1)  # 1*node_num

        return x


class CriticNetwork_GCN(torch.nn.Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(CriticNetwork_GCN, self).__init__()
        self.GCN1 = CensNet(node_in_features, edge_in_features, node_out_features,
                            edge_out_features)
        self.GCN2 = CensNet(node_out_features, edge_out_features,
                            node_out_features, edge_out_features)
        self.predict_v1 = torch.nn.Linear(node_out_features, node_out_features)
        self.predict_v2 = torch.nn.Linear(node_out_features, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T):
        """
        forward of both actor and critic
        """
        node, edge = self.GCN1(node, edge, node_adj, edge_adj, D_v, D_e, T)
        # node, edge = adopt_batch_norm(node, edge, self.b1, self.b2)
        node, edge = self.GCN2(node, edge, node_adj, edge_adj, D_v, D_e, T)
        # node, edge = adopt_batch_norm(node, edge, self.b3, self.b4)
        value = F.relu(self.predict_v1(node))  # 1*node_num*node_out_features
        value = torch.mean(value, dim=1)  # 1*node_out_features
        value = self.predict_v2(value)  # 1*1

        return value


class Edgethick_Actor(nn.Module):
    """入力がstate.
    """

    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(Edgethick_Actor, self).__init__()
        self.GCN1 = CensNet(node_in_features, edge_in_features, node_out_features,
                            edge_out_features)
        self.GCN2 = CensNet(node_out_features, edge_out_features,
                            node_out_features, edge_out_features)
        self.predict_v1 = torch.nn.Linear(node_out_features, node_out_features)
        self.mean = torch.nn.Linear(4 * node_out_features, 1)
        self.std = torch.nn.Linear(4 * node_out_features, 1)

        self.saved_actions = []

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T, node1, node2):
        """
        forward of both actor and critic
        """
        node, edge = self.GCN1(node, edge, node_adj, edge_adj, D_v, D_e, T)
        node, edge = self.GCN2(node, edge, node_adj, edge_adj, D_v, D_e, T)
        node = F.relu(self.predict_v1(node))  # 1*node_num*node_out_features
        node1_2 = node[:, [node1, node2], :]  # 1*2*node_out_features
        node1_2 = node.reshape(1, -1)  # 1*(2*node_out_features)

        # var to mean todekaeru
        mean = self.mean(node1_2)
        std = self.std(node1_2)
        mean = torch.abs(mean)
        std = torch.abs(std)
        y = torch.cat([mean, std], dim=1)

        return y


class X_Y_Actor(nn.Module):
    """入力がstate.
    """

    def __init__(self, node_in_features, emb_size):
        super(X_Y_Actor, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, emb_size)

        self.mean = torch.nn.Linear(emb_size, 2)
        self.std = torch.nn.Linear(emb_size, 2)

        self.saved_actions = []

    def forward(self, emb_node, node1):
        node1_emb = emb_node[0][node1]  # 1*node_in_features
        x = F.relu(self.layer1(node1_emb))  # 1*emb_size
        x = F.relu(self.layer2(x))  # 1*emb_size

        # var to mean todekaeru
        mean = self.mean(x)
        std = self.std(x)
        mean = torch.abs(mean)
        std = torch.abs(std)
        y = 10 * torch.cat([mean, std], dim=1)  # [mean[0],mean[1],std[0],std[1]]

        return y
