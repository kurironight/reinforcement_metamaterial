import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeCensNet(Module):

    def __init__(self, in_features, out_features, edge_features):
        super(NodeCensNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.node_weight = Parameter(
            torch.DoubleTensor(in_features, out_features))
        self.edge_weight = Parameter(torch.DoubleTensor(edge_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.edge_weight.size(0))
        self.edge_weight.data.uniform_(-stdv, stdv)

    def forward(self, node, edge, node_adj, D, T):
        # T: エッジとノードの連結行列を示すバイナリ行列
        # Av~を求める（この部分はdataloaderの前でもできる)
        # TODO Dがいらない疑惑
        D = torch.sqrt(D)  # DはN*N
        D[[D != 0]] = torch.pow(D[D != 0], -1)
        node_adj += torch.eye(node_adj.size(2)).double()  # 単位行列
        A_tilde = torch.bmm(D, node_adj)
        A_tilde = torch.bmm(node_adj, D)
        # fiを求める
        HePe = torch.matmul(edge, self.edge_weight)
        fi = torch.diag_embed(HePe)  # 対角行列より作成
        # T*fi*TT*Av~*Hv*Wvを行う
        output = torch.bmm(T, fi)
        output = torch.bmm(output, torch.transpose(T, 1, 2))
        output = output * A_tilde
        output = torch.bmm(output, node)
        output = torch.matmul(output, self.node_weight)

        return output


class EdgeCensNet(Module):

    def __init__(self, in_features, out_features, node_features):
        super(EdgeCensNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_features = node_features
        self.edge_weight = Parameter(
            torch.DoubleTensor(in_features, out_features))
        self.node_weight = Parameter(torch.DoubleTensor(node_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.node_weight.size(0))
        self.node_weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.edge_weight.size(1))
        self.edge_weight.data.uniform_(-stdv, stdv)

    def forward(self, node, edge, edge_adj, D, T):
        # T: エッジとノードの連結行列を示すバイナリ行列
        # Ae~を求める
        D = torch.sqrt(D)  # DはN*N
        D[[D != 0]] = torch.pow(D[D != 0], -1)
        edge_adj += torch.eye(edge_adj.size(2)).double()  # 単位行列
        A_tilde = torch.bmm(D, edge_adj)
        A_tilde = torch.bmm(edge_adj, D)
        # fiを求める
        HvPv = torch.matmul(node, self.node_weight)
        fi = torch.diag_embed(HvPv)  # 対角行列より作成
        # TT*fi*T*Ae~*e*HE*Weを行う
        output = torch.bmm(torch.transpose(T, 1, 2), fi)
        output = torch.bmm(output, T)
        output = output * A_tilde
        output = torch.bmm(output, edge)
        output = torch.matmul(output, self.edge_weight)

        return output


class CensNet(Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(CensNet, self).__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.node_out_features = node_out_features
        self.edge_out_features = edge_out_features
        self.nodenet = NodeCensNet(
            self.node_in_features, self.node_out_features,
            self.edge_in_features)
        self.edgenet = EdgeCensNet(
            self.edge_in_features, self.edge_out_features,
            self.node_out_features)
        self.lrelu1 = torch.nn.LeakyReLU()
        self.lrelu2 = torch.nn.LeakyReLU()

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T):
        # print("node:", node.shape)
        # print("edge:", edge.shape)
        # print("node_adj:", node_adj.shape)
        # print("edge_adj:", edge_adj.shape)
        # print("D_v:", D_v.shape)
        # print("D_e:", D_e.shape)
        # print("T:", T.shape)
        node = self.nodenet(node, edge, node_adj, D_v, T)
        node = self.lrelu1(node)
        edge = self.edgenet(node, edge, edge_adj, D_e, T)
        edge = self.lrelu2(edge)
        return node, edge
