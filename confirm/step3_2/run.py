from .condition import easy_dev
import torch.optim as optim
import torch
from tools.plot import plot_efficiency_history
import os
from .actor_critic import *
from env.gym_barfem import BarFemGym
import numpy as np
import pickle
from .examine_possibility_distribution import convert_Vp_to_edgethick
import matplotlib.pyplot as plt


def actor_gcn_critic_gcn(max_episodes=5000, test_name="test", log_file=False):
    """Actor-Criticを行う．Actor,CriticはGCN
    Actorの指定できるものは，一つのエッジのみの幅を選択できる．
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

    log_dir = "confirm/step3_2/a_gcn_c_gcn_results/{}".format(test_name)

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

    criticNet = CriticNetwork_GCN(2, 1, 400, 400).to(device).double()
    edgethickNet = Edgethick_Actor(2, 1, 400, 400).to(device).double()
    optimizer_edgethick = optim.Adam(edgethickNet.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in range(max_episodes):
        if log_file:
            with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
                print('\nepoch:', episode, file=f)
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        for step in range(max_steps):
            action = select_action_gcn_critic_gcn(
                env, criticNet, edgethickNet, device, log_dir=log_file, history=history)

            next_nodes_pos, _, done, _ = env.step(action)
            reward = env.calculate_simulation(mode='force')
            criticNet.rewards.append(reward)

        loss = finish_episode(criticNet, edgethickNet, optimizer_critic,
                              optimizer_edgethick, gamma, log_dir=log_file, history=history)

        history['epoch'].append(episode + 1)
        history['result_efficiency'].append(reward)

        if episode % 100 == 0:
            print("episode:{} total reward:{}".format(episode, reward))

    env.close()
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))

    return history


def actor_gcn_critic_gcn_mean(test_num=5, max_episodes=5000, test_name="test", log_file=None):
    """Actor-Criticの５回実験したときの平均グラフを作成する関数"""

    log_dir = "confirm/step3_2/a_gcn_c_gcn_results/{}".format(test_name)
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


def plot_Vp_mu_sigma_history(history, save_path):  # step3_2の時のmu,sigma,Vp_aの挙動をチェックする為の関数
    with open("confirm/step3_2/Vp_edgethick_set/edges_thicknesses.pkl", 'rb') as web:
        edge_thicknesses = pickle.load(web)
    with open("confirm/step3_2/Vp_edgethick_set/rewards.pkl", 'rb') as web:
        rewards = pickle.load(web)
        rewards = np.array(rewards).flatten()

    epochs = history['epoch']
    a_mean = np.array(history['a_mean'])
    a_sigma = np.array(history['a_sigma'])
    advantage = np.array(history['advantage'])
    a = np.array(history['a'])
    Vp_a = np.array([convert_Vp_to_edgethick(edge_thicknesses, rewards, Vp) for Vp in np.array(np.array(history['critic_value']))])
    result_efficiency = np.array(history['result_efficiency'])
    epochs = np.array(epochs)
    result_efficiency = np.array(result_efficiency)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(epochs, a_mean + a_sigma, a_mean - a_sigma, facecolor='y', alpha=0.5)

    ax.plot(epochs, Vp_a, label='Vp_a')
    ax.plot(epochs, a_mean, label='a_mean')
    ax.plot(epochs, a, label='a')
    ax.plot(epochs, advantage, label='advantage')
    #ax.set_xlim(0, 8000)
    ax.set_xlabel('epoch')
    #ax.set_ylim(0.15, 0.32)
    ax.legend()
    ax.set_title("edgethick curve")
    plt.savefig(save_path)
    plt.close()


def plot_Vp_efficiency_history(history, save_path):
    with open("confirm/step3_2/Vp_edgethick_set/edges_thicknesses.pkl", 'rb') as web:
        edge_thicknesses = pickle.load(web)
    with open("confirm/step3_2/Vp_edgethick_set/rewards.pkl", 'rb') as web:
        rewards = pickle.load(web)
        rewards = np.array(rewards).flatten()

    epochs = history['epoch']
    Vp = np.array(history['critic_value'])
    result_efficiency = np.array(history['result_efficiency'])
    epochs = np.array(epochs)
    result_efficiency = np.array(result_efficiency)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, Vp, label='Vp')
    ax.plot(epochs, result_efficiency, label='result_efficiency')
    ax.set_xlim(6000, 10000)
    ax.set_xlabel('epoch')
    ax.set_ylim(0.8, 1.3)
    ax.legend()
    ax.set_title("edgethick curve")
    plt.savefig(save_path)
    plt.close()
