
import pickle
import os
import numpy as np
from collections import namedtuple
from policy import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from env.gym_fem import FEMGym
from tools.graph import make_T_matrix, make_edge_adj, make_D_matrix
import torch.distributions as tdist
from tools.lattice_preprocess import make_continuous_init_graph
from tqdm import tqdm
from itertools import count
from tools.plot import plot_loss_history, plot_reward_history, plot_efficiency_history, plot_steps_history


# 初期のノードの状態を抽出
origin_nodes_positions = np.array([
    [0., 0.86603], [0.5, 0.], [1., 0.86603], [1.5, 0.],
    [2., 0.86603], [2.5, 0.], [3., 0.86603], [3.5, 0.],
    [4., 0.86603], [4.5, 0.], [5., 0.86603], [5.5, 0.],
    [6., 0.86603], [6.5, 0.], [7., 0.86603], [7.5, 0.],
    [8., 0.86603], [0., 2.59808], [0.5, 1.73205], [1., 2.59808],
    [1.5, 1.73205], [2., 2.59808], [2.5, 1.73205], [3., 2.59808],
    [3.5, 1.73205], [4., 2.59808], [4.5, 1.73205], [5., 2.59808],
    [5.5, 1.73205], [6., 2.59808], [6.5, 1.73205], [7., 2.59808],
    [7.5, 1.73205], [8., 2.59808], [0., 4.33013], [0.5, 3.4641],
    [1., 4.33013], [1.5, 3.4641], [2., 4.33013], [2.5, 3.4641],
    [3., 4.33013], [3.5, 3.4641], [4., 4.33013], [4.5, 3.4641],
    [5., 4.33013], [5.5, 3.4641], [6., 4.33013], [6.5, 3.4641],
    [7., 4.33013], [7.5, 3.4641], [8., 4.33013], [0., 6.06218],
    [0.5, 5.19615], [1., 6.06218], [1.5, 5.19615], [2., 6.06218],
    [2.5, 5.19615], [3., 6.06218], [3.5, 5.19615], [4., 6.06218],
    [4.5, 5.19615], [5., 6.06218], [5.5, 5.19615], [6., 6.06218],
    [6.5, 5.19615], [7., 6.06218], [7.5, 5.19615], [8., 6.06218],
    [0., 8], [0.5, 6.9282], [1., 8], [1.5, 6.9282],
    [2., 8], [2.5, 6.9282], [3., 8], [3.5, 6.9282],
    [4., 8], [4.5, 6.9282], [5., 8], [5.5, 6.9282],
    [6., 8], [6.5, 6.9282], [7., 8], [7.5, 6.9282],
    [8., 8]])

origin_nodes_positions = origin_nodes_positions/8

origin_edges_indices = np.array([
    [0,  1], [0,  2], [0, 18], [1,  2], [1,  3], [2, 18],
    [2,  3], [2,  4], [2, 20], [3,  5], [3,  4], [4,  5],
    [4, 22], [4, 20], [4,  6], [5,  7], [5,  6], [6, 22],
    [6,  7], [6, 24], [6,  8], [7,  9], [7,  8], [8, 10],
    [8, 24], [8,  9], [8, 26], [9, 10], [9, 11], [10, 26],
    [10, 11], [10, 12], [10, 28], [11, 12], [11, 13], [12, 28],
    [12, 13], [12, 14], [12, 30], [13, 15], [13, 14], [14, 15],
    [14, 32], [14, 30], [14, 16], [15, 16], [16, 32], [17, 19],
    [17, 18], [17, 35], [18, 19], [18, 20], [19, 35], [19, 21],
    [19, 20], [19, 37], [20, 22], [20, 21], [21, 22], [21, 39],
    [21, 37], [21, 23], [22, 24], [22, 23], [23, 24], [23, 39],
    [23, 41], [23, 25], [24, 26], [24, 25], [25, 27], [25, 26],
    [25, 41], [25, 43], [26, 27], [26, 28], [27, 43], [27, 29],
    [27, 28], [27, 45], [28, 29], [28, 30], [29, 45], [29, 31],
    [29, 30], [29, 47], [30, 32], [30, 31], [31, 32], [31, 49],
    [31, 47], [31, 33], [32, 33], [33, 49], [34, 35], [34, 36],
    [34, 52], [35, 36], [35, 37], [36, 52], [36, 37], [36, 38],
    [36, 54], [37, 39], [37, 38], [38, 39], [38, 56], [38, 54],
    [38, 40], [39, 41], [39, 40], [40, 56], [40, 41], [40, 58],
    [40, 42], [41, 43], [41, 42], [42, 44], [42, 58], [42, 43],
    [42, 60], [43, 44], [43, 45], [44, 60], [44, 45], [44, 46],
    [44, 62], [45, 46], [45, 47], [46, 62], [46, 47], [46, 48],
    [46, 64], [47, 49], [47, 48], [48, 49], [48, 66], [48, 64],
    [48, 50], [49, 50], [50, 66], [51, 53], [51, 52], [51, 69],
    [52, 53], [52, 54], [53, 69], [53, 55], [53, 54], [53, 71],
    [54, 56], [54, 55], [55, 56], [55, 73], [55, 71], [55, 57],
    [56, 58], [56, 57], [57, 58], [57, 73], [57, 75], [57, 59],
    [58, 60], [58, 59], [59, 61], [59, 60], [59, 75], [59, 77],
    [60, 61], [60, 62], [61, 77], [61, 63], [61, 62], [61, 79],
    [62, 63], [62, 64], [63, 79], [63, 65], [63, 64], [63, 81],
    [64, 66], [64, 65], [65, 66], [65, 83], [65, 81], [65, 67],
    [66, 67], [67, 83], [68, 69], [68, 70], [69, 70], [69, 71],
    [70, 71], [70, 72], [71, 73], [71, 72], [72, 73], [72, 74],
    [73, 75], [73, 74], [74, 75], [74, 76], [75, 77], [75, 76],
    [76, 78], [76, 77], [77, 78], [77, 79], [78, 79], [78, 80],
    [79, 80], [79, 81], [80, 81], [80, 82], [81, 83], [81, 82],
    [82, 83], [82, 84], [83, 84],
])

# origin_edges_indices = np.concatenate(
#    [origin_edges_indices, [[81, 68], [68, 9]]])
origin_input_nodes = [84]
origin_input_vectors = np.array([
    [0., -0.1],
    [0., -0.1],
    [0., -0.1],
    [0., -0.1]
])

origin_output_nodes = [68]
origin_output_vectors = np.array([
    [-1, 0],
    [-1, 0],
    [-1, 0],
    [-1, 0],
])

origin_frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]


# パラメータ
test_name = "test3"  # 実験名
node_out_features = 5
node_features = 3  # 座標2つ，ラベル1つ.変わらない値．
gamma = 0.99  # 割引率
lr = 0.03  # 学習率
train_num = 20  # 学習回数
max_action = 500  # 1episodeの最大試行回数
penalty = 0.001  # 連続状態から不連続状態になった時のペナルティ
final_penalty = 2  # 時刻内に終了しなかった場合のペナルティ
continuous_reward = 2  # 連続状態に初めてなったときにあげる報酬

EDGE_THICKNESS = 0.2  # エッジの太さ
log_dir = "results/{}".format(test_name)

assert not os.path.exists(log_dir), "already folder exists"
os.makedirs(log_dir)

# 学習の推移
history = {}
history['epoch'] = []
history['loss'] = []
history['ep_reward'] = []
history['result_efficiency'] = []
history['steps'] = []


# モデル定義
GCN = model.GCN_fund_model(node_features, 1, node_out_features, 3).double()
X_Y = model.X_Y_model(node_out_features, 2).double()  # なぜかdoubleが必要だった
Stop = model.Stop_model(node_out_features, 2).double()
Select_node1 = model.Select_node1_model(node_out_features, 2).double()
Select_node2 = model.Select_node2_model(
    node_features+node_out_features, 2).double()
Edge_thickness = model.Edge_thickness_model(
    node_features+node_out_features, 2).double()


# 連続状態を作成
while(1):
    new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness = make_continuous_init_graph(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                                                                                                                                                                                  origin_output_nodes, origin_output_vectors, origin_frozen_nodes, EDGE_THICKNESS)
    env = FEMGym(new_node_pos,
                 new_edges_indices, new_edges_thickness)
    env.reset()
    if env.confirm_graph_is_connected():
        env.render('render_image/yatta.png')
        break

Saved_Action = namedtuple('SavedAction', ['action', 'value'])
Saved_prob_Action = namedtuple('SavedAction', ['log_prob'])
Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance'])
GCN_optimizer = optim.Adam(GCN.parameters(), lr=lr)
X_Y_optimizer = optim.Adam(X_Y.parameters(), lr=lr)
Stop_optimizer = optim.Adam(Stop.parameters(), lr=lr)
Select_node1_optimizer = optim.Adam(Select_node1.parameters(), lr=lr)
Select_node2_optimizer = optim.Adam(Select_node2.parameters(), lr=lr)
Edge_thickness_optimizer = optim.Adam(Edge_thickness.parameters(), lr=lr)


def select_action(first_node_num):
    nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()

    node_num = nodes_pos.shape[0]

    # ラベル作成
    label = np.zeros((node_num, 1))
    label[:first_node_num] = 1
    nodes_pos = np.concatenate([nodes_pos, label], 1)

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

    # action求め
    emb_graph, state_value = GCN(node, edge, node_adj, edge_adj, D_v, D_e, T)
    # 新規ノードの座標決め
    coord = X_Y(emb_graph)
    coord_x_tdist = tdist.Normal(coord[0][0].item(), coord[0][2].item())
    coord_y_tdist = tdist.Normal(coord[0][1].item(), coord[0][3].item())
    coord_x_action = coord_x_tdist.sample()
    coord_y_action = coord_y_tdist.sample()
    # 0-1に収める
    coord_x_action = torch.clamp(coord_x_action, min=0, max=1)
    coord_y_action = torch.clamp(coord_y_action, min=0, max=1)

    # グラフ追加を止めるかどうか
    stop_prob = Stop(emb_graph)
    stop_categ = Categorical(stop_prob)
    stop = stop_categ.sample()

    # ノード1を求める
    node1_prob = Select_node1(emb_graph)
    node1_categ = Categorical(node1_prob)
    node1 = node1_categ.sample()

    # 新規ノード
    new_node = torch.Tensor([coord_x_action, coord_y_action, 0]).double()
    new_node = torch.reshape(new_node, (1, 1, 3))
    node_cat = torch.cat([node, new_node], 1)

    # H1を除いたnode_catの作成
    non_node1_node_cat = torch.cat(
        [node_cat[:, 0:node1, :], node_cat[:, node1+1:, :]], 1)

    # H1の情報抽出
    H1 = emb_graph[0][node1]
    H1_cat = H1.repeat(non_node1_node_cat.shape[1], 1)
    H1_cat = H1_cat.unsqueeze(0)
    # HとH1のノード情報をconcat
    emb_graph_cat = torch.cat([non_node1_node_cat, H1_cat], 2)

    # ノード2を求める
    node2_prob = Select_node2(emb_graph_cat)
    node2_categ = Categorical(node2_prob)
    node2_temp = node2_categ.sample()
    if node2_temp >= node1:
        node2 = node2_temp+1  # node1分の調整
    else:
        node2 = node2_temp
    H2 = node_cat[0][node2]  # node_posよりH2の特徴を抽出

    # エッジの太さ決め
    # H1とH2のノード情報をconcat
    H1 = H1.unsqueeze(0)
    H2 = H2.unsqueeze(0)
    emb_graph_cat2 = torch.cat([H1, H2], 2)
    edge_thickness = Edge_thickness(emb_graph_cat2)
    edge_thickness_tdist = tdist.Normal(
        edge_thickness[0][0].item(), edge_thickness[0][1].item())
    edge_thickness_action = edge_thickness_tdist.sample()
    edge_thickness_action = torch.clamp(edge_thickness_action, min=0, max=1)

    action = {}
    action["which_node"] = np.array(
        [node1.item(), node2.item()])
    action["new_node"] = np.array([[coord_x_action, coord_y_action]])
    action["edge_thickness"] = np.array(
        [edge_thickness_action])
    action["end"] = stop.item()

    # save to action buffer
    GCN.saved_actions.append(Saved_Action(action, state_value))
    Stop.saved_actions.append(Saved_prob_Action(
        stop_categ.log_prob(stop)))
    Select_node1.saved_actions.append(Saved_prob_Action(
        node1_categ.log_prob(node1)))
    Select_node2.saved_actions.append(Saved_prob_Action(
        node2_categ.log_prob(node2_temp)))  # node2_tempを利用していることに注意

    X_Y.saved_actions.append(Saved_mean_std_Action(
        coord[0][:2], coord[0][2:]))
    Edge_thickness.saved_actions.append(Saved_mean_std_Action(
        edge_thickness[0][0], edge_thickness[0][1]))
    GCN.node_nums.append(node_num)

    return action


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    GCN_saved_actions = GCN.saved_actions
    X_Y_saved_actions = X_Y.saved_actions
    Stop_saved_actions = Stop.saved_actions
    Select_node1_saved_actions = Select_node1.saved_actions
    Select_node2_saved_actions = Select_node2.saved_actions
    Edge_thickness_saved_actions = Edge_thickness.saved_actions

    node_nums = GCN.node_nums

    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in GCN.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)

    for (action, value), (x_y_mean, x_y_var), stop_prob, node1_prob, node2_prob, (edge_thick_mean, edge_thick_var), node_num, R in zip(GCN_saved_actions, X_Y_saved_actions,
                                                                                                                                       Stop_saved_actions, Select_node1_saved_actions,
                                                                                                                                       Select_node2_saved_actions, Edge_thickness_saved_actions,
                                                                                                                                       node_nums, returns):

        advantage = R - value.item()

        # calculate actor (policy) loss
        if action["end"]:
            policy_losses.append(-stop_prob.log_prob[0] * advantage)
        else:
            log_probs = torch.cat([stop_prob.log_prob,
                                   node1_prob.log_prob, node2_prob.log_prob])
            if advantage <= 0:
                policy_losses.append(-torch.mean(log_probs) * advantage)
            else:
                # 新規ノードが選択されているとき
                if np.isin(node_num, action["which_node"]):
                    x_y_mean_loss = F.l1_loss(torch.from_numpy(
                        action["new_node"][0]).double(), x_y_mean)
                    edge_thick_mean_loss = F.l1_loss(torch.from_numpy(
                        action["edge_thickness"]).double(), edge_thick_mean)

                    x_y_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["new_node"][0]-x_y_mean.to('cpu').detach().numpy().copy())).double(), torch.sqrt(x_y_var))
                    edge_thick_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["edge_thickness"]-edge_thick_mean.item())).double(), torch.sqrt(edge_thick_var))

                    mean_loss = (x_y_mean_loss*2+edge_thick_mean_loss)/3
                    var_loss = (x_y_var_loss*2+edge_thick_var_loss)/3

                    policy_losses.append(
                        (-torch.mean(log_probs)+mean_loss+var_loss) * advantage)
                else:
                    edge_thick_mean_loss = F.l1_loss(torch.from_numpy(
                        action["edge_thickness"]).double(), torch.tensor([edge_thick_mean]))
                    edge_thick_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["edge_thickness"]-edge_thick_mean.item())).double(), torch.tensor([torch.sqrt(edge_thick_var)]))
                    policy_losses.append(
                        (-torch.mean(log_probs)+edge_thick_mean_loss+edge_thick_var_loss) * advantage)

        # calculate critic (value) loss using L1 loss
        value_losses.append(F.l1_loss(value, torch.tensor([[R]]).double()))

    # reset gradients
    GCN_optimizer.zero_grad()
    X_Y_optimizer.zero_grad()
    Stop_optimizer.zero_grad()
    Select_node1_optimizer.zero_grad()
    Select_node2_optimizer.zero_grad()
    Edge_thickness_optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    GCN_optimizer.step()
    X_Y_optimizer.step()
    Stop_optimizer.step()
    Select_node1_optimizer.step()
    Select_node2_optimizer.step()
    Edge_thickness_optimizer.step()

    # reset rewards and action buffer
    del GCN.rewards[:]
    del GCN.saved_actions[:]
    del GCN.node_nums[:]
    del X_Y.saved_actions[:]
    del Stop.saved_actions[:]
    del Select_node1.saved_actions[:]
    del Select_node2.saved_actions[:]
    del Edge_thickness.saved_actions[:]

    return loss.item()


def save_model(save_name="Good"):
    torch.save(GCN.state_dict(), os.path.join(
        log_dir, '{}_GCN.pth'.format(save_name)))
    torch.save(X_Y.state_dict(), os.path.join(
        log_dir, '{}_X_Y.pth'.format(save_name)))
    torch.save(Stop.state_dict(), os.path.join(
        log_dir, '{}_Stop.pth'.format(save_name)))
    torch.save(Select_node1.state_dict(), os.path.join(
        log_dir, '{}_Select_node1.pth'.format(save_name)))
    torch.save(Select_node2.state_dict(), os.path.join(
        log_dir, '{}_Select_node2.pth'.format(save_name)))
    torch.save(Edge_thickness.state_dict(), os.path.join(
        log_dir, '{}_Edge_thickness.pth'.format(save_name)))


def main():
    # running_reward = 0
    prior_efficiency = 0
    continuous_trigger = 0

    best_efficiency = -1000
    best_epoch = 0

    # １エピソードのループ
    while(1):
        new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness = make_continuous_init_graph(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                                                                                                                                                                                      origin_output_nodes, origin_output_vectors, origin_frozen_nodes, EDGE_THICKNESS)
        env = FEMGym(new_node_pos,
                     new_edges_indices, new_edges_thickness)
        env.reset()
        if env.confirm_graph_is_connected():
            break
    nodes_pos, _, _, _ = env.extract_node_edge_info()
    first_node_num = nodes_pos.shape[0]

    # run inifinitely many episodes
    for epoch in tqdm(range(train_num)):
        # for epoch in count(1):

        # reset environment and episode reward
        while(1):
            new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness = make_continuous_init_graph(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                                                                                                                                                                                          origin_output_nodes, origin_output_vectors, origin_frozen_nodes, EDGE_THICKNESS)
            env = FEMGym(new_node_pos,
                         new_edges_indices, new_edges_thickness)
            env.reset()
            if env.confirm_graph_is_connected():
                break
        state = env.reset()
        ep_reward = 0
        continuous_trigger = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(max_action):
            # select action from policy
            action = select_action(first_node_num)
            nodes_pos, edges_indices, edges_thickness, adj = env.extract_node_edge_info()

            # take the action
            state, _, done, info = env.step(action)
            if (t == (max_action-1)) and (done is not True):  # max_action内にてactionが終わらない時
                reward = -final_penalty
            elif env.confirm_graph_is_connected():
                efficiency = env.calculate_simulation()
                if continuous_trigger == 1:
                    reward = efficiency-prior_efficiency
                else:
                    reward = efficiency+continuous_reward
                    continuous_trigger = 1
                prior_efficiency = efficiency

            elif continuous_trigger == 1:
                reward = -penalty
            else:
                reward = 0

            GCN.rewards.append(reward)

            ep_reward += reward
            if done:
                steps = t
                break

        # update cumulative reward
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        loss = finish_episode()

        # efficiencyの最終結果を求める
        if env.confirm_graph_is_connected():
            result_efficiency = env.calculate_simulation()
        else:
            result_efficiency = -1

        if best_efficiency < result_efficiency:
            best_epoch = epoch
            best_efficiency = result_efficiency
            save_model(save_name="Good")
            env.render(os.path.join(
                log_dir, 'render_image/{}.png'.format(epoch+1)))

        history['epoch'].append(epoch+1)
        history['loss'].append(loss)
        history['ep_reward'].append(ep_reward)
        history['result_efficiency'].append(result_efficiency)
        history['steps'].append(steps+1)

        # 学習履歴を保存
        with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
            f.writelines('epoch %d, loss: %.4f ep_reward: %.4f result_efficiency: %.4f\n' %
                         (epoch + 1, loss, ep_reward, result_efficiency))
        with open(os.path.join(log_dir, "represent_value.txt"), mode='w') as f:
            f.writelines('epoch %d,  best_efficiency: %.4f\n' %
                         (best_epoch+1, best_efficiency))
        save_model(save_name="Last")

        plot_loss_history(history, os.path.join(
            log_dir, 'learning_loss_curve.png'))
        plot_reward_history(history, os.path.join(
            log_dir, 'learning_reward_curve.png'))
        plot_efficiency_history(history, os.path.join(
            log_dir, 'learning_effi_curve.png'))
        plot_steps_history(history, os.path.join(
            log_dir, 'learning_steps_curve.png'))


if __name__ == '__main__':
    main()
