import optuna
from .step2_gym import Step2Gym
from .condition import easy_dev
import numpy as np
import torch.optim as optim
import torch
from tools.plot import plot_efficiency_history
import os
from .actor_critic import select_action, ActorNetwork, ActorNetwork2, finish_episode, CriticNetwork
import pickle


def actor_critic():
    """Actor-Criticを行う．学習はDL（GCN以外）で
    Actorの指定できる幅は0.1-1となっている"""

    max_episodes = 5000
    test_name = "5000"  # 実験名

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    log_dir = "confirm/step2/ac_results/{}".format(test_name)
    # assert not os.path.exists(log_dir), "already folder exists"
    os.makedirs(log_dir, exist_ok=True)

    node_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes = easy_dev()
    env = Step2Gym(node_pos, input_nodes, input_vectors,
                   output_nodes, output_vectors, frozen_nodes,
                   edges_indices, edges_thickness, frozen_nodes)
    env.reset()

    max_steps = 1
    lr_actor = 1e-4
    lr_critic = 1e-3
    weight_decay = 1e-2
    gamma = 0.99

    device = torch.device('cpu')

    actorNet = ActorNetwork().to(device)
    actorNet2 = ActorNetwork2().to(device)
    criticNet = CriticNetwork().to(device)
    optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
    optimizer_actor2 = optim.Adam(actorNet2.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in range(max_episodes):
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        for step in range(max_steps):
            action = select_action(
                nodes_pos, actorNet, actorNet2, criticNet, device)

            next_nodes_pos, _, done, _ = env.step(action)
            reward = env.calculate_simulation(mode='force')
            criticNet.rewards.append(reward)

        loss = finish_episode(criticNet, actorNet, actorNet2, optimizer_critic,
                              optimizer_actor, optimizer_actor2, gamma)

        history['epoch'].append(episode+1)
        history['result_efficiency'].append(reward)

        if episode % 10 == 0:
            print("episode:{} total reward:{}".format(episode, reward))

    env.close()
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))


"""




def actor_critic_mean():

    test_num = 5

    max_episodes = 5000
    test_name = "5timess_with_std"  # 実験名

    log_dir = "confirm/step2/ac_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
    os.makedirs(log_dir, exist_ok=True)

    history = {}
    for i in range(test_num):
        history["{}".format(i)] = {}
        history["{}".format(i)]['epoch'] = []
        history["{}".format(i)]['result_efficiency'] = []

        node_pos, input_nodes, input_vectors,\
            output_nodes, output_vectors, frozen_nodes,\
            edges_indices, edges_thickness, frozen_nodes = easy_dev()
        env = Step1Gym(node_pos, input_nodes, input_vectors,
                       output_nodes, output_vectors, frozen_nodes,
                       edges_indices, edges_thickness, frozen_nodes)
        env.reset()

        max_steps = 1
        lr_actor = 1e-4
        lr_critic = 1e-3
        weight_decay = 1e-2
        gamma = 0.99

        device = torch.device('cpu')

        actorNet = Edgethick_Actor().to(device)
        criticNet = Edgethick_Critic().to(device)
        optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
        optimizer_critic = optim.Adam(
            criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

        for episode in range(max_episodes):
            observation = env.reset()
            observation = np.array([0, 1])

            for step in range(max_steps):
                action = select_action(
                    observation, actorNet, criticNet, device)

                next_observation, _, done, _ = env.step(action)
                reward = env.calculate_simulation(mode='force')
                criticNet.rewards.append(reward)

            loss = finish_episode(criticNet, actorNet, optimizer_critic,
                                  optimizer_actor, gamma)

            history["{}".format(i)]['epoch'].append(episode+1)
            history["{}".format(i)]['result_efficiency'].append(reward)

            if episode % 1000 == 0:
                print("episode:{} total reward:{}".format(episode, reward))

        env.close()
        plot_efficiency_history(history["{}".format(i)], os.path.join(
            log_dir, 'learning_effi_curve{}.png'.format(i)))

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
"""