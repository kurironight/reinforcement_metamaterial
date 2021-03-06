
import pickle
import os
import numpy as np
from collections import namedtuple
from policy import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from env.gym_barfem import BarFemGym
from tools.graph import make_T_matrix, make_edge_adj, make_D_matrix
import torch.distributions as tdist
from tqdm import tqdm
from tools.plot import plot_loss_history, plot_reward_history, plot_efficiency_history, plot_steps_history
from tools.save import load_graph_info

# パラメータ
test_name = "confirm_loss_contribution"  # 実験名
annealing_name = "barfem_50000"
node_out_features = 5
node_features = 3  # 座標2つ，ラベル1つ.変わらない値．
gamma = 0.99  # 割引率
lr = 0.03  # 学習率
train_num = 50000  # 学習回数
max_action = 500  # 1episodeの最大試行回数
penalty = 0.001  # 連続状態から不連続状態になった時のペナルティ
final_penalty = 2  # 時刻内に終了しなかった場合のペナルティ
continuous_reward = 2  # 連続状態に初めてなったときにあげる報酬
annealing_dir = "prior_thesis_results/{}".format(annealing_name)

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

# アニーリングの情報を引き継ぐ．
node_pos, input_nodes, input_vectors, output_nodes,\
    output_vectors, frozen_nodes, edges_indices, edges_thickness = load_graph_info(
        os.path.join(annealing_dir, 'graph_info'))
env = BarFemGym(node_pos, input_nodes, input_vectors,
                output_nodes, output_vectors, frozen_nodes,
                edges_indices, edges_thickness, frozen_nodes)
env.reset()


def select_action(condition_nodes):
    nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()

    node_num = nodes_pos.shape[0]

    # ラベル作成
    label = np.zeros((node_num, 1))
    label[condition_nodes] = 1
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


def finish_episode(epoch):
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
    with open(os.path.join(log_dir, "loss.txt"), mode='a') as f:
        f.writelines('epoch %d\n' % (epoch + 1))
    steps = 0

    for (action, value), (x_y_mean, x_y_var), stop_prob, node1_prob, node2_prob, (edge_thick_mean, edge_thick_var), node_num, R in zip(GCN_saved_actions, X_Y_saved_actions,
                                                                                                                                       Stop_saved_actions, Select_node1_saved_actions,
                                                                                                                                       Select_node2_saved_actions, Edge_thickness_saved_actions,
                                                                                                                                       node_nums, returns):

        # calculate critic (value) loss using L1 loss
        value_loss = F.l1_loss(value, torch.tensor([[R]]).double())
        value_losses.append(value_loss)

        steps += 1
        advantage = R - value.item()

        # calculate actor (policy) loss
        if action["end"]:
            policy_losses.append(-stop_prob.log_prob[0] * advantage)
            with open(os.path.join(log_dir, "loss.txt"), mode='a') as f:
                f.writelines('steps %d, value_loss: %.4f advantage: %.4f stop_prob: %.4f action: end\n' %
                             (steps, value_loss, advantage, stop_prob.log_prob[0]))
        else:
            log_probs = torch.cat([stop_prob.log_prob,
                                   node1_prob.log_prob, node2_prob.log_prob])
            if advantage <= 0:
                policy_losses.append(-torch.mean(log_probs) * advantage)
                with open(os.path.join(log_dir, "loss.txt"), mode='a') as f:
                    f.writelines('steps %d, value_loss: %.4f advantage: %.4f stop_prob: %.4f node1_prob: %.4f node2_prob: %.4f action: adv<0\n' %
                                 (steps, value_loss, advantage, stop_prob.log_prob[0], node1_prob.log_prob[0], node2_prob.log_prob[0]))
            else:
                if np.isin(node_num, action["which_node"]):  # 新規ノードが選択されているとき
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
                    with open(os.path.join(log_dir, "loss.txt"), mode='a') as f:
                        f.writelines('steps %d, value_loss: %.4f advantage: %.4f stop_prob: %.4f node1_prob: %.4f node2_prob: %.4f x_y_mean_loss: %.4f edge_thick_mean_loss: %.4f x_y_var_loss: %.4f edge_thick_var_loss: %.4f action: new_node\n\n\n' %
                                     (steps, value_loss, advantage, stop_prob.log_prob[0], node1_prob.log_prob[0], node2_prob.log_prob[0], x_y_mean_loss, edge_thick_mean_loss, x_y_var_loss, edge_thick_var_loss))
                else:
                    edge_thick_mean_loss = F.l1_loss(torch.from_numpy(
                        action["edge_thickness"]).double(), torch.tensor([edge_thick_mean]))
                    edge_thick_var_loss = F.l1_loss(torch.from_numpy(
                        np.abs(action["edge_thickness"]-edge_thick_mean.item())).double(), torch.tensor([torch.sqrt(edge_thick_var)]))
                    policy_losses.append(
                        (-torch.mean(log_probs)+edge_thick_mean_loss+edge_thick_var_loss) * advantage)
                    with open(os.path.join(log_dir, "loss.txt"), mode='a') as f:
                        f.writelines('steps %d, value_loss: %.4f advantage: %.4f stop_prob: %.4f node1_prob: %.4f node2_prob: %.4f edge_thick_mean_loss: %.4f edge_thick_var_loss: %.4f action: exist_node\n' %
                                     (steps, value_loss, advantage, stop_prob.log_prob[0], node1_prob.log_prob[0], node2_prob.log_prob[0], edge_thick_mean_loss, edge_thick_var_loss))

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

    # run inifinitely many episodes
    for epoch in tqdm(range(train_num)):
        # for epoch in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        continuous_trigger = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(max_action):
            # select action from policy
            action = select_action(env.condition_nodes)
            nodes_pos, edges_indices, edges_thickness, adj = env.extract_node_edge_info()

            # take the action
            state, _, done, info = env.step(action)
            if (t == (max_action-1)) and (done is not True):  # max_action内にてactionが終わらない時
                reward = np.array([-final_penalty])
            elif env.confirm_graph_is_connected():
                efficiency = env.calculate_simulation()
                if continuous_trigger == 1:
                    reward = efficiency-prior_efficiency
                else:
                    reward = efficiency+continuous_reward
                    continuous_trigger = 1
                prior_efficiency = efficiency

            elif continuous_trigger == 1:
                reward = np.array([-penalty])
            else:
                reward = np.array([0])

            GCN.rewards.append(reward)

            ep_reward += reward
            if done:
                steps = t
                break

        # update cumulative reward
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        loss = finish_episode(epoch)

        # efficiencyの最終結果を求める
        if env.confirm_graph_is_connected():
            result_efficiency = env.calculate_simulation()
        else:
            result_efficiency = -1

        if best_efficiency < result_efficiency:
            best_epoch = epoch
            best_efficiency = result_efficiency
            save_model(save_name="Good")
            # env.render(os.path.join(
            #    log_dir, 'render_image/{}.png'.format(epoch+1)))

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
