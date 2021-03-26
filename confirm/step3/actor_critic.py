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

Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance'])
Saved_Action = namedtuple('SavedAction', ['action', 'value'])
Saved_prob_Action = namedtuple('SavedAction', ['log_prob'])


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


class ActorNetwork_GCN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, node_out_features,
                 edge_out_features):
        super(ActorNetwork_GCN, self).__init__()
        self.GCN1 = CensNet(node_in_features, edge_in_features, node_out_features,
                            edge_out_features)
        self.GCN2 = CensNet(node_out_features, edge_out_features,
                            node_out_features, edge_out_features)
        self.predict_v1 = torch.nn.Linear(node_out_features, node_out_features)
        self.predict_v2 = torch.nn.Linear(node_out_features, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, node, edge, node_adj, edge_adj, D_v, D_e, T, remove_index=False):
        """
        forward of both actor and critic
        """
        node, edge = self.GCN1(node, edge, node_adj, edge_adj, D_v, D_e, T)
        # node, edge = adopt_batch_norm(node, edge, self.b1, self.b2)
        node, edge = self.GCN2(node, edge, node_adj, edge_adj, D_v, D_e, T)
        # node, edge = adopt_batch_norm(node, edge, self.b3, self.b4)
        node = F.relu(self.predict_v1(node))  # 1*node_num*node_out_features
        node = self.predict_v2(node)  # 1*node_num
        node = node.reshape((-1, node.size(1)))
        if remove_index is not False:
            node = torch.cat(
                [node[:, 0:remove_index], node[:, remove_index+1:]], 1)
        node = F.softmax(node, dim=-1)

        return node


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
        self.mean = torch.nn.Linear(4*node_out_features, 1)
        self.std = torch.nn.Linear(4*node_out_features, 1)

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
        mean = 0.1+0.9*torch.sigmoid(self.mean(node1_2))
        std = 0.1*torch.sigmoid(self.std(node1_2))
        y = torch.cat([mean, std], dim=1)

        return y


def make_torch_type_for_GCN(nodes_pos, edges_indices, edges_thickness, node_adj):
    # GCNの為の引数を作成
    T = make_T_matrix(edges_indices)
    edge_adj = make_edge_adj(edges_indices, T)
    D_e = make_D_matrix(edge_adj)
    D_v = make_D_matrix(node_adj)

    # GCNへの変換
    node = torch.from_numpy(nodes_pos).clone().double()
    node = node.unsqueeze(0)
    edge = torch.from_numpy(edges_thickness).clone().double()
    edge = edge.unsqueeze(0).unsqueeze(2)
    node_adj = torch.from_numpy(node_adj).clone().double()
    node_adj = node_adj.unsqueeze(0)
    edge_adj = torch.from_numpy(edge_adj).clone().double()
    edge_adj = edge_adj.unsqueeze(0)
    D_v = torch.from_numpy(D_v).clone().double()
    D_v = D_v.unsqueeze(0)
    D_e = torch.from_numpy(D_e).clone().double()
    D_e = D_e.unsqueeze(0)
    T = torch.from_numpy(T).clone().double()
    T = T.unsqueeze(0)

    return node, edge, node_adj, edge_adj, D_v, D_e, T


def select_action_gcn_critic_gcn(env, node1Net, node2Net, criticNet, edgethickNet, device):
    nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
    node, edge, node_adj, edge_adj, D_v, D_e, T = make_torch_type_for_GCN(
        nodes_pos, edges_indices, edges_thickness, node_adj)
    state_value = criticNet(node, edge, node_adj,
                            edge_adj, D_v, D_e, T)
    node1_prob = node1Net(node, edge, node_adj,
                          edge_adj, D_v, D_e, T)
    node1_categ = Categorical(node1_prob)
    node1 = node1_categ.sample()
    label = torch.zeros(1, 4, 1).double()
    label[:, node1, :] = 1

    node_labeled = torch.cat([node, label], 2)
    node2_prob = node2Net(node_labeled, edge, node_adj,
                          edge_adj, D_v, D_e, T, remove_index=node1)
    node2_categ = Categorical(node2_prob)
    node2_temp = node2_categ.sample()

    if node2_temp >= node1:
        node2 = node2_temp+1  # node1分の調整
    else:
        node2 = node2_temp
    edge_thickness = edgethickNet(node, edge, node_adj,
                                  edge_adj, D_v, D_e, T, node1, node2)
    edge_thickness_tdist = tdist.Normal(
        edge_thickness[0][0].item(), edge_thickness[0][1].item())
    edge_thickness_action = edge_thickness_tdist.sample()
    edge_thickness_action = torch.clamp(edge_thickness_action, min=0, max=1)

    action = {}
    action['which_node'] = np.array([node1.item(), node2.item()])
    action['end'] = 0
    action['edge_thickness'] = np.array([edge_thickness_action.item()])
    action['new_node'] = np.array([[0, 2]])

    # save to action buffer
    criticNet.saved_actions.append(Saved_Action(action, state_value))
    node1Net.saved_actions.append(Saved_prob_Action(
        node1_categ.log_prob(node1)))
    node2Net.saved_actions.append(Saved_prob_Action(
        node2_categ.log_prob(node2_temp)))
    edgethickNet.saved_actions.append(Saved_mean_std_Action(
        edge_thickness[0][0], edge_thickness[0][1]))

    return action


def finish_episode(Critic, node1Net, node2Net, edgethickNet, Critic_opt, Node1_opt, Node2_opt, Edge_thick_opt, gamma):
    R = 0
    GCN_saved_actions = Critic.saved_actions
    node1Net_saved_actions = node1Net.saved_actions
    node2Net_saved_actions = node2Net.saved_actions
    Edge_thickness_saved_actions = edgethickNet.saved_actions

    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in Critic.rewards[:: -1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    for (action, value), (edge_thick_mean, edge_thick_std), node1_prob, node2_prob,   R in zip(GCN_saved_actions, Edge_thickness_saved_actions, node1Net_saved_actions, node2Net_saved_actions, returns):

        advantage = R - value.item()
        # advantage = advantage.to(torch.float)  # なぜかfloatにしないとエラーを吐いた．
        # print("advantage:", advantage)

        # calculate actor (policy) loss
        if action["end"]:
            print("okasii")
        else:
            log_probs = torch.cat(
                [node1_prob.log_prob, node2_prob.log_prob])
            policy_loss = -torch.mean(log_probs) * advantage

            policy_losses.append(policy_loss)
            if advantage > 0:
                edge_thick_mean_loss = F.l1_loss(torch.from_numpy(
                    action["edge_thickness"]).double(), edge_thick_mean.reshape((1)).double())
                edge_thick_var_loss = F.l1_loss(torch.from_numpy(
                    np.abs(action["edge_thickness"]-edge_thick_mean.item())).double(), edge_thick_std.reshape((1)).double())
                policy_losses.append(
                    (edge_thick_mean_loss+edge_thick_var_loss) * advantage)

        # calculate critic (value) loss using L1 loss
        value_losses.append(
            F.l1_loss(value.double(), torch.tensor([[R]]).double()))

    # print("policy_losses:", policy_losses)
    # print("value_losses:", value_losses)

    # reset gradients
    Critic_opt.zero_grad()
    Node1_opt.zero_grad()
    Node2_opt.zero_grad()
    Edge_thick_opt.zero_grad()

    # sum up all the values of policy_losses and value_losses
    if len(policy_losses) == 0:
        loss = torch.stack(value_losses).sum()
    else:
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    Critic_opt.step()
    Node1_opt.step()
    Node2_opt.step()
    Edge_thick_opt.step()

    # reset rewards and action buffer
    del Critic.rewards[:]
    del Critic.saved_actions[:]
    del node1Net.saved_actions[:]
    del node2Net.saved_actions[:]
    del edgethickNet.saved_actions[:]

    return loss.item()
