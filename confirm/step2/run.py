import optuna
from .step2_gym import Step2Gym
from .condition import easy_dev
import numpy as np
import torch.optim as optim
import torch
from tools.plot import plot_efficiency_history
import os
from .actor_critic import *
import pickle


def actor_critic():
    """Actor-Criticを行う．学習はDL（GCN以外）で
    Actorの指定できるものは，ノード1とノード2であり，一つのエッジのみを選択できる．"""

    max_episodes = 5000
    test_name = "5000"  # 実験名

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    log_dir = "confirm/step2/ac_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
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


def actor_critic_gcn():
    """Actor-Criticを行う．CriticはGCN
    Actorの指定できるものは，ノード1とノード2であり，一つのエッジのみを選択できる．"""

    max_episodes = 5000
    test_name = "test"  # 実験名

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    log_dir = "confirm/step2/ac_gcn_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
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
    criticNet = CriticNetwork_GCN(2, 1, 400, 400).to(device).double()
    optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
    optimizer_actor2 = optim.Adam(actorNet2.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in range(max_episodes):
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        for step in range(max_steps):
            action = select_action_critic_gcn(
                env, actorNet, actorNet2, criticNet, device)

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


def actor_gcn_critic_gcn():
    """Actor-Criticを行う．Actor,CriticはGCN
    Actorの指定できるものは，ノード1とノード2であり，一つのエッジのみを選択できる．"""

    max_episodes = 5000
    test_name = "test"  # 実験名

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    log_dir = "confirm/step2/a_gcn_c_gcn_results/{}".format(test_name)
    #assert not os.path.exists(log_dir), "already folder exists"
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

    actorNet = ActorNetwork_GCN(2, 1, 400, 400).to(device).double()
    actorNet2 = ActorNetwork_GCN(3, 1, 400, 400).to(device).double()
    criticNet = CriticNetwork_GCN(2, 1, 400, 400).to(device).double()
    optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
    optimizer_actor2 = optim.Adam(actorNet2.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)

    for episode in range(max_episodes):
        env.reset()
        nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()
        for step in range(max_steps):
            action = select_action_gcn_critic_gcn(
                env, actorNet, actorNet2, criticNet, device)

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
