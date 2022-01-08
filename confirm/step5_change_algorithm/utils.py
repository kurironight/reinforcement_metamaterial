import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tools.graph import make_T_matrix, make_edge_adj, make_D_matrix
from collections import namedtuple
import torch.distributions as tdist

Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance'])
Saved_Action = namedtuple('SavedAction', ['action', 'value'])
Saved_prob_Action = namedtuple('SavedAction', ['log_prob'])


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


def select_action_gcn_critic_gcn(env, criticNet, node1Net, node2Net, x_y_Net, device, log_dir=None, history=None):
    nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
    node_num = nodes_pos.shape[0]
    node, edge, node_adj, edge_adj, D_v, D_e, T = make_torch_type_for_GCN(
        nodes_pos, edges_indices, edges_thickness, node_adj)
    state_value = criticNet(node, edge, node_adj,
                            edge_adj, D_v, D_e, T)

    # ノード1を求める
    emb_node, node1_prob = node1Net(node, edge, node_adj,
                                    edge_adj, D_v, D_e, T)
    node1_categ = Categorical(node1_prob)
    node1 = node1_categ.sample()

    # 新規ノードの座標求め
    x_y = x_y_Net(emb_node, node1)
    x_tdist = tdist.Normal(
        x_y[0][0].item(), x_y[0][2].item())
    y_tdist = tdist.Normal(
        x_y[0][1].item(), x_y[0][3].item())
    x_action = x_tdist.sample()
    x_action = torch.clamp(x_action, min=0, max=1)
    y_action = y_tdist.sample()
    y_action = torch.clamp(y_action, min=0, max=1)

    # 新規ノード
    new_node = torch.Tensor([x_action.item(), y_action.item()]).double()
    new_node = torch.reshape(new_node, (1, 1, 2))
    node_cat = torch.cat([node, new_node], 1)

    # ノード1を除いたnode_catの作成
    non_node1_node_cat = torch.cat(
        [node_cat[:, 0:node1, :], node_cat[:, node1 + 1:, :]], 1)

    # ノード1の情報抽出
    H1 = emb_node[0][node1]
    H1_cat = H1.repeat(node_num, 1)
    H1_cat = H1_cat.unsqueeze(0)
    # HとH1のノード情報をconcat
    emb_graph_cat = torch.cat([non_node1_node_cat, H1_cat], 2)

    # ノード2を求める
    node2_prob = node2Net(emb_graph_cat)
    node2_categ = Categorical(node2_prob)
    node2_temp = node2_categ.sample()
    if node2_temp >= node1:
        node2 = node2_temp + 1  # node1分の調整
    else:
        node2 = node2_temp

    action = {}
    action['which_node'] = np.array([node1.item(), node2.item()])
    action['end'] = 0
    action['edge_thickness'] = np.array([1])
    action['new_node'] = np.array([[x_action.item(), y_action.item()]])

    # save to action buffer
    criticNet.saved_actions.append(Saved_Action(action, state_value))
    x_y_Net.saved_actions.append(Saved_mean_std_Action(
        x_y[0][:2], x_y[0][2:]))
    node1Net.saved_actions.append(Saved_prob_Action(
        node1_categ.log_prob(node1)))
    node2Net.saved_actions.append(Saved_prob_Action(
        node2_categ.log_prob(node2_temp)))

    if history is not None:
        # historyにログを残す
        history['x'].append(x_action.item())
        history['x_mean'].append(x_y[0][0].item())
        history['x_sigma'].append(x_y[0][2].item())
        history['y'].append(y_action.item())
        history['y_mean'].append(x_y[0][1].item())
        history['y_sigma'].append(x_y[0][3].item())
        history['critic_value'].append(state_value.item())
        history['node_select'].append([node1.item(), node2.item()])
        history['node1_possibility'].append(node1_prob.to('cpu').detach().numpy().copy())
        history['node2_possibility'].append(node2_prob.to('cpu').detach().numpy().copy())
    return action


def finish_episode(Critic, x_y_Net, node1Net, node2Net, Critic_opt, x_y_opt, Node1_opt, Node2_opt, gamma, log_dir=None, history=None):
    R = 0
    GCN_saved_actions = Critic.saved_actions
    x_y_saved_actions = x_y_Net.saved_actions
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
                if 4 in action['which_node']:
                    x_y_mean_loss = F.l1_loss(torch.from_numpy(
                        action["new_node"][0]).double(), x_y_mean.double())
                    x_y_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["new_node"][0] - x_y_mean.to('cpu').detach().numpy().copy())), x_y_std.double())
                    policy_losses.append((x_y_mean_loss + x_y_var_loss) * advantage)
                    x_y_opt_trigger = True  # x_y_optのトリガーを起動

        # calculate critic (value) loss using L1 loss
        value_losses.append(
            F.l1_loss(value.double(), torch.tensor([[R]]).double()))

    # reset gradients
    Critic_opt.zero_grad()
    Node1_opt.zero_grad()
    Node2_opt.zero_grad()
    if x_y_opt_trigger:
        x_y_opt.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + \
        torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    Critic_opt.step()
    Node1_opt.step()
    Node2_opt.step()
    if x_y_opt_trigger:
        x_y_opt.step()

    # reset rewards and action buffer
    del Critic.rewards[:]
    del Critic.saved_actions[:]
    del x_y_Net.saved_actions[:]
    del node1Net.saved_actions[:]
    del node2Net.saved_actions[:]

    if history is not None:
        history['advantage'].append(advantage.item())
    return loss.item()
