from itertools import islice
import random
from collections import deque, namedtuple
import copy
import gym
import pickle
import os
import numpy as np
from collections import namedtuple
from policy import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributions as tdist

Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance'])
Saved_Action = namedtuple('SavedAction', ['action', 'value'])
Saved_prob_Action = namedtuple('SavedAction', ['log_prob'])


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


class ActorNetwork(nn.Module):
    def __init__(self, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(2, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

        self.saved_actions = []

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        y = y.reshape((1, 4))
        y = F.softmax(y, dim=-1)
        return y


class ActorNetwork2(nn.Module):
    def __init__(self, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(ActorNetwork2, self).__init__()
        self.fc1 = nn.Linear(2, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

        self.saved_actions = []

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        y = y.reshape((1, 3))
        y = F.softmax(y, dim=-1)
        return y


class CriticNetwork(nn.Module):
    def __init__(self,  hidden1_size=400, hidden2_size=300, init_w=3e-4):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(2, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)
        self.fc4 = nn.Linear(4, 1)

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = F.relu(self.fc3(h))
        y = y.reshape((1, 4))
        y = self.fc4(y)

        return y


def select_action(state, node1Net, node2Net, criticNet, device):
    "GCNを用いない手法"
    state_tensor = torch.tensor(
        state, dtype=torch.float, device=device).view(1, 4, 2)
    state_value = criticNet(state_tensor)
    node1_prob = node1Net(state_tensor)
    node1_categ = Categorical(node1_prob)
    node1 = node1_categ.sample()
    non_node1_state = torch.cat(
        [state_tensor[:, 0:node1, :], state_tensor[:, node1+1:, :]], 1)
    node2_prob = node2Net(non_node1_state)
    node2_categ = Categorical(node2_prob)
    node2_temp = node2_categ.sample()

    if node2_temp >= node1:
        node2 = node2_temp+1  # node1分の調整
    else:
        node2 = node2_temp

    action = {}
    action['which_node'] = np.array([node1.item(), node2.item()])
    action['end'] = 0
    action['edge_thickness'] = np.array([1])
    action['new_node'] = np.array([[0, 2]])

    # save to action buffer
    criticNet.saved_actions.append(Saved_Action(action, state_value))
    node1Net.saved_actions.append(Saved_prob_Action(
        node1_categ.log_prob(node1)))
    node2Net.saved_actions.append(Saved_prob_Action(
        node2_categ.log_prob(node2_temp)))

    return action


def finish_episode(Critic, node1Net, node2Net, Critic_opt, Node1_opt, Node2_opt, gamma):
    R = 0
    GCN_saved_actions = Critic.saved_actions
    node1Net_saved_actions = node1Net.saved_actions
    node2Net_saved_actions = node2Net.saved_actions

    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in Critic.rewards[:: -1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    for (action, value),  node1_prob, node2_prob,   R in zip(GCN_saved_actions, node1Net_saved_actions, node2Net_saved_actions, returns):

        advantage = R - value.item()
        advantage = advantage.to(torch.float)
        # print("advantage:", advantage)

        # calculate actor (policy) loss
        if action["end"]:
            print("okasii")
        else:
            log_probs = torch.cat(
                [node1_prob.log_prob, node2_prob.log_prob])
            policy_loss = -torch.mean(log_probs) * advantage

            policy_losses.append(policy_loss)

        # calculate critic (value) loss using L1 loss
        value_losses.append(
            F.l1_loss(value.double(), torch.tensor([[R]]).double()))

    # print("policy_losses:", policy_losses)
    # print("value_losses:", value_losses)

    # reset gradients
    Critic_opt.zero_grad()
    Node1_opt.zero_grad()
    Node2_opt.zero_grad()

    # sum up all the values of policy_losses and value_losses
    if len(policy_losses) == 0:
        loss = torch.stack(value_losses).sum()
    else:
        # loss = torch.stack(policy_losses).sum() + \
        #     torch.stack(value_losses).sum()
        loss = torch.stack(policy_losses).sum()

    # perform backprop
    loss.backward()
    Critic_opt.step()
    Node1_opt.step()
    Node2_opt.step()

    # reset rewards and action buffer
    del Critic.rewards[:]
    del Critic.saved_actions[:]
    del node1Net.saved_actions[:]
    del node2Net.saved_actions[:]

    return loss.item()
