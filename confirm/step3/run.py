from .condition import easy_dev
import torch.optim as optim
import torch
from tools.plot import plot_efficiency_history
import os
from .actor_critic import *
from env.gym_barfem import BarFemGym
import numpy as np
import pickle
from tqdm import tqdm


def confirm_max_status():
    "最大値となる状態を求める．"
    max = 0
    x = 1000
    for i in np.arange(0.1, 1.001, 0.001):
        node_pos, input_nodes, input_vectors,\
            output_nodes, output_vectors, frozen_nodes,\
            edges_indices, edges_thickness, frozen_nodes = easy_dev()
        env = BarFemGym(node_pos, input_nodes, input_vectors,
                        output_nodes, output_vectors, frozen_nodes,
                        edges_indices, edges_thickness, frozen_nodes)
        env.reset()

        action = {}
        action['which_node'] = np.array([0, 3])
        action['end'] = 0
        action['edge_thickness'] = np.array([i])
        action['new_node'] = np.array([[0, 2]])

        next_nodes_pos, _, done, _ = env.step(action)
        reward = env.calculate_simulation()
        if max < reward:
            max = reward
            x = i
    print("最小は", x, max)


def actor_gcn_critic_gcn(max_episodes=5000, test_name="test", log_file=False, save_pth=False):
    """Actor-Criticを行う．Actor,CriticはGCN
    Actorの指定できるものは，ノード1とノード2であり，一つのエッジのみを選択できる．
    max_episodes:学習回数
    test_name:保存ファイルの名前
    log_file: Trueにすると，progress.txtに損失関数などの情報のログをとる．"""

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []
    history['mean_efficiency'] = []  # a_meanの値の時のηの値を収納する
    history['a'] = []
    history['a_mean'] = []
    history['a_sigma'] = []
    history['advantage'] = []
    history['critic_value'] = []

    log_dir = "confirm/step3/a_gcn_c_gcn_results/{}".format(test_name)

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

    max_steps = 1
    lr_actor = 1e-4
    lr_critic = 1e-3
    weight_decay = 1e-2
    gamma = 0.99

    device = torch.device('cpu')

    actorNet = Select_node1_model(2, 1, 400, 400).to(device).double()
    actorNet2 = Select_node2_model(400 + 2, 400).to(device).double()
    criticNet = CriticNetwork_GCN(2, 1, 400, 400).to(device).double()
    edgethickNet = Edgethick_Actor(2, 1, 400, 400).to(device).double()
    optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
    optimizer_actor2 = optim.Adam(actorNet2.parameters(), lr=lr_actor)
    optimizer_edgethick = optim.Adam(edgethickNet.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in tqdm(range(max_episodes)):
        if log_file:
            with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
                print('\nepoch:', episode, file=f)
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        for step in range(max_steps):
            action = select_action_gcn_critic_gcn(
                env, actorNet, actorNet2, criticNet, edgethickNet, device, log_dir=log_file)

            next_nodes_pos, _, done, _ = env.step(action)
            reward = env.calculate_simulation()
            criticNet.rewards.append(reward)

        loss = finish_episode(criticNet, actorNet, actorNet2, edgethickNet, optimizer_critic,
                              optimizer_actor, optimizer_actor2, optimizer_edgethick, gamma, log_dir=log_file)

        history['epoch'].append(episode + 1)
        history['result_efficiency'].append(reward)
        if episode % 100 == 0:
            if save_pth:
                save_model(criticNet, edgethickNet, os.path.join(log_dir, "pth"), save_name=str(episode))

    env.close()
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))

    return history


def actor_gcn_critic_gcn_mean(test_num=5, max_episodes=5000, test_name="test", log_file=None):
    """Actor-Criticの５回実験したときの平均グラフを作成する関数"""

    log_dir = "confirm/step3/a_gcn_c_gcn_results/{}".format(test_name)
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
