import torch
import torch.nn as nn
from .utils import set_init, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import math
import os
from .condition import easy_dev
from .actor_critic import *
from env.gym_barfem import BarFemGym
import numpy as np
from confirm.step5_multi.actor_critic import Select_node1_model, Select_node2_model, CriticNetwork_GCN, Edgethick_Actor, X_Y_Actor


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        # entropy=m.entropy() これでも可
        entropy = 0.5 + 0.5 * \
            math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_criticNet, global_x_y_Net, global_node1Net, global_node2Net,
                 Critic_opt, x_y_opt, Node1_opt, Node2_opt, global_ep, global_ep_r, res_queue, name, gamma=0.99, total_episodes=5000):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.Critic_opt, self.x_y_opt, self.Node1_opt, self.Node2_opt = Critic_opt, x_y_opt, Node1_opt, Node2_opt
        self.global_criticNet, self.global_x_y_Net,\
            self.global_node1Net, self.global_node2Net = global_criticNet, global_x_y_Net, global_node1Net, global_node2Net
        device = torch.device('cpu')
        self.local_criticNet = CriticNetwork_GCN(2, 1, 400, 400).double().to(device)
        self.local_x_y_Net = X_Y_Actor(2, 1, 400, 400).double().to(device)
        self.local_node1Net = Select_node1_model(2, 1, 400, 400).double().to(device)
        self.local_node2Net = Select_node2_model(400 + 2, 400).double().to(device)

        node_pos, input_nodes, input_vectors,\
            output_nodes, output_vectors, frozen_nodes,\
            edges_indices, edges_thickness, frozen_nodes = easy_dev()
        self.env = BarFemGym(node_pos, input_nodes, input_vectors,
                             output_nodes, output_vectors, frozen_nodes,
                             edges_indices, edges_thickness, frozen_nodes)
        self.env.reset()

        self.gamma = gamma  # 報酬減衰率
        self.total_episodes = total_episodes  # すべてのプロセスにおいての合計epoch

    def finish_episode(self, log_dir=None, history=None):
        R = 0
        GCN_saved_actions = self.local_criticNet.saved_actions
        x_y_saved_actions = self.local_x_y_Net.saved_actions
        node1Net_saved_actions = self.local_node1Net.saved_actions
        node2Net_saved_actions = self.local_node2Net.saved_actions

        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.local_criticNet.rewards[:: -1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        x_y_opt_trigger = False  # advantage>0の場合したときにx_y_optを作動出来るようにする為のトリガー
        for (action, value), (x_y_mean, x_y_std), node1_prob, node2_prob, R in zip(GCN_saved_actions, x_y_saved_actions, node1Net_saved_actions, node2Net_saved_actions, returns):

            advantage = R - value.item()

            # calculate actor (policy) loss
            if action["end"]:
                print("okasii")
            else:
                log_probs = torch.cat(
                    [node1_prob.log_prob, node2_prob.log_prob])
                policy_loss = -torch.mean(log_probs) * advantage

                policy_losses.append(policy_loss)
                if advantage > 0:
                    x_y_mean_loss = F.l1_loss(torch.from_numpy(
                        action["new_node"][0]).double(), x_y_mean.double())
                    x_y_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["new_node"][0] - x_y_mean.to('cpu').detach().numpy().copy())), x_y_std.double())
                    policy_losses.append((x_y_mean_loss + x_y_var_loss) * advantage)

                    x_y_opt_trigger = True  # x_y_optのトリガーを起動
                else:
                    x_y_mean_loss = torch.zeros(1)
                    x_y_var_loss = torch.zeros(1)

            # calculate critic (value) loss using L1 loss
            value_losses.append(
                F.l1_loss(value.double(), torch.tensor([[R]]).double()))

        # reset gradients
        self.Critic_opt.zero_grad()
        self.Node1_opt.zero_grad()
        self.Node2_opt.zero_grad()
        if x_y_opt_trigger:
            self.x_y_opt.zero_grad()

        # sum up all the values of policy_losses and value_losses
        if len(policy_losses) == 0:
            loss = torch.stack(value_losses).sum()
        else:
            loss = torch.stack(policy_losses).sum() + \
                torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        for lp, gp in zip(self.local_criticNet.parameters(), self.global_criticNet.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.local_node1Net.parameters(), self.global_node1Net.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.local_node2Net.parameters(), self.global_node2Net.parameters()):
            gp._grad = lp.grad
        if x_y_opt_trigger:
            for lp, gp in zip(self.local_x_y_Net.parameters(), self.global_x_y_Net.parameters()):
                gp._grad = lp.grad

        self.Critic_opt.step()
        self.Node1_opt.step()
        self.Node2_opt.step()
        if x_y_opt_trigger:
            self.x_y_opt.step()

        # pull global parameters
        self.local_criticNet.load_state_dict(self.global_criticNet.state_dict())
        self.local_x_y_Net.load_state_dict(self.global_x_y_Net.state_dict())
        self.local_node1Net.load_state_dict(self.global_node1Net.state_dict())
        self.local_node2Net.load_state_dict(self.global_node2Net.state_dict())

        # reset rewards and action buffer
        del self.local_criticNet.rewards[:]
        del self.local_criticNet.saved_actions[:]
        del self.local_x_y_Net.saved_actions[:]
        del self.local_node1Net.saved_actions[:]
        del self.local_node2Net.saved_actions[:]

        if history is not None:
            history['advantage'].append(advantage.item())
        return loss.item()

    def run(self, max_episodes=5000, test_name="test", log_file=False, save_pth=False, history=None, device=torch.device('cpu')):
        while self.g_ep.value < self.total_episodes:
            # 入力ノードを再設定している為，ここに追加
            node_pos, input_nodes, input_vectors,\
                output_nodes, output_vectors, frozen_nodes,\
                edges_indices, edges_thickness, frozen_nodes = easy_dev()
            self.env = BarFemGym(node_pos, input_nodes, input_vectors,
                                 output_nodes, output_vectors, frozen_nodes,
                                 edges_indices, edges_thickness, frozen_nodes)
            self.env.reset()
            for episode in range(max_episodes):
                action = select_action_gcn_critic_gcn(
                    self.env, self.local_criticNet, self.local_node1Net, self.local_node2Net, self.local_x_y_Net, device, log_dir=log_file, history=history)
                next_nodes_pos, _, done, _ = self.env.step(action)
                if 4 in action['which_node']:
                    self.env.input_nodes = [2, 4]
                    self.env.input_vectors = np.array([[1, 0], [0, 1]])
                if 2 in action['which_node'] and 4 in action['which_node']:  # TODO [2,4]を選択しないように学習させる
                    reward = 0
                else:
                    reward = self.env.calculate_simulation()
                self.local_criticNet.rewards.append(reward)

                done = True  # 今回はonestepの為

                if done:  # update global and assign to local net
                    record(self.g_ep, self.g_ep_r, reward,
                           self.res_queue, self.name)
                    # sync
                    # 各プロセスの重み更新をglobalにpushし，その更新後のものを各プロセスの重みに戻す
                    self.finish_episode()

                    break

        self.res_queue.put(None)
