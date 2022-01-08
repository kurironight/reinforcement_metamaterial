from .condition import easy_dev
import torch.optim as optim
import torch
from tools.plot import plot_efficiency_history
import os
from .actor_critic import *
from .utils import select_action_gcn_critic_gcn, finish_episode
from env.gym_barfem import BarFemGym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def actor_gcn_critic_gcn(max_episodes=5000, test_name="test", log_file=False, save_pth=False):
    """Actor-Criticを行う．Actor,CriticはGCN
    Actorの指定できるものは，一つのエッジのみの幅を選択できる．
    max_episodes:学習回数
    test_name:保存ファイルの名前
    log_file: Trueにすると，progress.txtに損失関数などの情報のログをとる．"""

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []
    history['x'] = []
    history['x_mean'] = []
    history['x_sigma'] = []
    history['y'] = []
    history['y_mean'] = []
    history['y_sigma'] = []
    history['advantage'] = []
    history['critic_value'] = []
    history['node_select'] = []
    history['node1_possibility'] = []
    history['node2_possibility'] = []

    log_dir = "confirm/step5_change_algorithm/a_gcn_c_gcn_results/{}".format(test_name)

    assert not os.path.exists(log_dir), "already folder exists"
    if log_file:
        log_file = log_dir
    else:
        log_file = None
    os.makedirs(log_dir, exist_ok=True)

    node_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes = easy_dev()
    env = BarFemGym(node_pos, input_nodes, input_vectors,
                    output_nodes, output_vectors, frozen_nodes,
                    edges_indices, edges_thickness, frozen_nodes)
    env.reset()

    lr_actor = 1e-4
    lr_critic = 1e-3
    weight_decay = 1e-2
    gamma = 0.99

    device = torch.device('cpu')

    criticNet = CriticNetwork_GCN(2, 1, 400, 400).to(device).double()
    x_y_Net = X_Y_Actor(400, 400).to(device).double()
    node1Net = Select_node1_model(2, 1, 400, 400).to(device).double()
    node2Net = Select_node2_model(400 + 2, 400).to(device).double()  # 400+2における400は，Select_node1_modelのinput3の部分に対応
    optimizer_node1 = optim.Adam(node1Net.parameters(), lr=lr_actor)
    optimizer_node2 = optim.Adam(node2Net.parameters(), lr=lr_actor)
    optimizer_xy = optim.Adam(x_y_Net.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in tqdm(range(max_episodes)):
        if log_file:
            with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
                print('\nepoch:', episode, file=f)
        env = BarFemGym(node_pos, input_nodes, input_vectors,
                        output_nodes, output_vectors, frozen_nodes,
                        edges_indices, edges_thickness, frozen_nodes)
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        action = select_action_gcn_critic_gcn(
            env, criticNet, node1Net, node2Net, x_y_Net, device, log_dir=log_file, history=history)
        next_nodes_pos, _, done, _ = env.step(action)
        if 4 in action['which_node']:
            env.input_nodes = [2, 4]
            env.input_vectors = np.array([[1, 0], [0, 1]])
        if 2 in action['which_node'] and 4 in action['which_node']:  # TODO [2,4]を選択しないように学習させる
            reward = np.array([0])
        else:
            reward = env.calculate_simulation()
        criticNet.rewards.append(reward)

        loss = finish_episode(criticNet, x_y_Net, node1Net, node2Net,
                              optimizer_critic, optimizer_xy, optimizer_node1, optimizer_node2, gamma, log_dir=log_file, history=history)

        history['epoch'].append(episode + 1)
        history['result_efficiency'].append(reward)
        plot_efficiency_history(history, os.path.join(
            log_dir, 'learning_effi_curve.png'))
        if episode % 100 == 0:
            if save_pth:
                save_model(criticNet, x_y_Net, os.path.join(log_dir, "pth"), save_name=str(episode))

    env.close()
    with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    return history


def actor_gcn_critic_gcn_mean(test_num=5, max_episodes=5000, test_name="test", log_file=None):
    """Actor-Criticの５回実験したときの平均グラフを作成する関数"""

    log_dir = "confirm/step5_change_algorithm/a_gcn_c_gcn_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
    os.makedirs(log_dir, exist_ok=True)

    history = {}
    for i in range(test_num):
        history["{}".format(i)] = actor_gcn_critic_gcn(max_episodes=max_episodes, test_name=os.path.join(test_name, str(i)), log_file=log_file)

    mean = np.stack([history["{}".format(i)]['result_efficiency']
                     for i in range(test_num)])
    std = np.std(mean[:, -1])
    print('最終結果の標準偏差：', std)
    mean = np.mean(mean, axis=0)

    meanhistory = {}
    meanhistory['epoch'] = history['0']['epoch']
    meanhistory['result_efficiency'] = mean

    # 学習履歴を保存
    with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    plot_efficiency_history(meanhistory, os.path.join(
        log_dir, 'mean_learning_effi_curve.png'))


def save_model(criticNet, edgethickNet, log_dir, save_name="Good"):
    os.makedirs(log_dir, exist_ok=True)
    torch.save(criticNet.state_dict(), os.path.join(
        log_dir, '{}_criticNet.pth'.format(save_name)))
    torch.save(edgethickNet.state_dict(), os.path.join(
        log_dir, '{}_edgethickNet.pth'.format(save_name)))
