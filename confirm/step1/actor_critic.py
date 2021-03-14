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
from .DDPG import init_weight, ActorNetwork, CriticNetwork

Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance'])
Saved_Action = namedtuple('SavedAction', ['action', 'value'])


class Edgethick_Actor(ActorNetwork):
    def __init__(self, emb_size=300, hidden2_size=300):
        super(Edgethick_Actor, self).__init__(num_state=emb_size, num_action=2,
                                              hidden1_size=400, hidden2_size=hidden2_size, init_w=3e-3)
        self.fc3 = nn.Linear(hidden2_size, 1)
        self.fc4 = nn.Linear(hidden2_size, 1)
        self.saved_actions = []

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mean = torch.sigmoid(self.fc3(h))  # var to mean todekaeru
        std = 0.1*torch.sigmoid(self.fc4(h))
        y = torch.cat([mean, std], dim=1)
        print(y)
        return y


class Edgethick_Critic(CriticNetwork):
    def __init__(self, hidden1_size=400, hidden2_size=300):
        super(Edgethick_Critic, self).__init__(num_state=2, num_action=1,
                                               hidden1_size=hidden1_size, hidden2_size=hidden2_size, init_w=3e-4)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return h, y


def select_action(state, actorNet, criticNet, device):
    state_tensor = torch.tensor(
        state, dtype=torch.float, device=device).view(-1, 2)
    emb_graph, state_value = criticNet(state_tensor)
    edge_thickness = actorNet(emb_graph)
    edge_thickness_tdist = tdist.Normal(
        edge_thickness[0][0].item(), edge_thickness[0][1].item())
    edge_thickness_action = edge_thickness_tdist.sample()
    edge_thickness_action = torch.clamp(edge_thickness_action, min=0.1, max=1)

    action = {}
    action['which_node'] = np.array([0, 1])
    action['end'] = 0
    action['edge_thickness'] = np.array([edge_thickness_action.item()])
    action['new_node'] = np.array([[0, 2]])

    # save to action buffer
    criticNet.saved_actions.append(Saved_Action(action, state_value))
    actorNet.saved_actions.append(Saved_mean_std_Action(
        edge_thickness[0][0], edge_thickness[0][1]))

    return action


def finish_episode(Critic, Edge_thickness, Critic_opt, Edge_thick_opt, gamma):
    R = 0
    GCN_saved_actions = Critic.saved_actions
    Edge_thickness_saved_actions = Edge_thickness.saved_actions

    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in Critic.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    for (action, value),  (edge_thick_mean, edge_thick_std),  R in zip(GCN_saved_actions, Edge_thickness_saved_actions, returns):

        advantage = R - value.item()
        node_num = 2
        print("advantage:", advantage)

        # calculate actor (policy) loss
        if action["end"]:
            print("okasii")
        else:
            if advantage > 0:
                # 新規ノードが選択されているとき
                if np.isin(node_num, action["which_node"]):
                    print("okasii")
                else:
                    edge_thick_mean_loss = F.l1_loss(torch.from_numpy(
                        action["edge_thickness"]).double(), edge_thick_mean.reshape((1)).double())
                    edge_thick_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["edge_thickness"]-edge_thick_mean.item())).double(), edge_thick_std.reshape((1)).double())
                    policy_losses.append(
                        (edge_thick_mean_loss+edge_thick_var_loss) * advantage)

        # calculate critic (value) loss using L1 loss
        value_losses.append(
            F.l1_loss(value.double(), torch.tensor([[R]]).double()))

    #print("policy_losses:", policy_losses)
    #print("value_losses:", value_losses)

    # reset gradients
    Critic_opt.zero_grad()
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
    Edge_thick_opt.step()

    # reset rewards and action buffer
    del Critic.rewards[:]
    del Critic.saved_actions[:]
    del Edge_thickness.saved_actions[:]

    return loss.item()
