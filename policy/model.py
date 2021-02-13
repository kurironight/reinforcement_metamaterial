import torch
from GCN.layer import CensNet
import torch.nn.functional as F


def adopt_batch_norm(node, edge, b1, b2):
    node = node.permute(0, 2, 1)  # BatchNormを適用するための手順
    edge = edge.permute(0, 2, 1)  # BatchNormを適用するための手順
    node = b1(node)
    edge = b2(edge)
    node = node.permute(0, 2, 1)
    edge = edge.permute(0, 2, 1)

    return node, edge


class GCN_fund_model(torch.nn.Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(GCN_fund_model, self).__init__()
        self.GCN1 = CensNet(node_in_features, edge_in_features, node_out_features,
                            edge_out_features)
        self.GCN2 = CensNet(node_out_features, edge_out_features,
                            node_out_features, edge_out_features)
        self.predict_v1 = torch.nn.Linear(node_out_features, node_out_features)
        self.predict_v2 = torch.nn.Linear(node_out_features, 1)

        self.b1 = torch.nn.BatchNorm1d(node_out_features)
        self.b2 = torch.nn.BatchNorm1d(edge_out_features)
        self.b3 = torch.nn.BatchNorm1d(node_out_features)
        self.b4 = torch.nn.BatchNorm1d(edge_out_features)

        self.saved_actions = []
        self.rewards = []
        self.node_nums = []

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T):
        """
        forward of both actor and critic
        """
        node, edge = self.GCN1(node, edge, node_adj, edge_adj, D_v, D_e, T)
        node, edge = adopt_batch_norm(node, edge, self.b1, self.b2)
        node, edge = self.GCN2(node, edge, node_adj, edge_adj, D_v, D_e, T)
        node, edge = adopt_batch_norm(node, edge, self.b3, self.b4)
        value = F.relu(self.predict_v1(node))  # 1*node_num*node_out_features
        value = torch.mean(value, dim=1)  # 1*node_out_features
        value = self.predict_v2(value)  # 1*1

        return node, value


class X_Y_model(torch.nn.Module):
    def __init__(self, node_in_features, emb_size):
        super(X_Y_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, 4)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_graph):
        x = F.relu(self.layer1(emb_graph))  # 1*node_num*emb_size
        x = torch.mean(x, dim=1)  # 1*emb_size
        x = torch.sigmoid(self.layer2(x))  # 1*4

        return x


class Stop_model(torch.nn.Module):
    def __init__(self, node_in_features, emb_size):
        super(Stop_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, 2)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_graph):
        x = F.relu(self.layer1(emb_graph))  # 1*node_num*emb_size
        x = torch.mean(x, dim=1)  # 1*emb_size
        x = F.softmax(self.layer2(x), dim=-1)  # 1*2
        return x


class Select_node1_model(torch.nn.Module):
    def __init__(self, node_in_features, emb_size):
        super(Select_node1_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, emb_size)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_graph):
        x = self.layer1(emb_graph)  # 1*node_num*emb_size
        x = self.layer2(x)  # 1*node_num*emb_size
        x = F.softmax(torch.mean(x, dim=1), dim=-1)  # 1*node_num

        return x


class Select_node2_model(torch.nn.Module):
    # TODO ノード1以外を選ぶように指定することが出来ていない
    def __init__(self, node_in_features, emb_size):
        super(Select_node2_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, emb_size)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_graph):
        x = self.layer1(emb_graph)  # 1*node_num*emb_size
        x = self.layer2(x)  # 1*node_num*emb_size
        x = F.softmax(torch.mean(x, dim=1), dim=-1)  # 1*node_num

        return x


class Edge_thickness_model(torch.nn.Module):
    def __init__(self, node_in_features, emb_size):
        super(Edge_thickness_model, self).__init__()
        self.layer1 = torch.nn.Linear(node_in_features, emb_size)
        self.layer2 = torch.nn.Linear(emb_size, 2)

        # action & reward buffer
        self.saved_actions = []

    def forward(self, emb_graph):
        x = F.relu(self.layer1(emb_graph))  # 1*2*emb_size
        x = torch.mean(x, dim=1)  # 1*emb_size
        x = torch.sigmoid(self.layer2(x))  # 1*2

        return x
