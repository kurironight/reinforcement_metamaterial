import optuna
from .step1_gym import Step1Gym
from .condition import easy_dev
import numpy as np
import torch.optim as optim
import torch
from .DDPG import ActorNetwork, CriticNetwork, ReplayBuffer, DDPG
from tools.plot import plot_efficiency_history
import os
from .actor_critic import select_action, Edgethick_Actor, Edgethick_Critic, finish_episode


def f(x):
    node_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes = easy_dev()
    env = Step1Gym(node_pos, input_nodes, input_vectors,
                   output_nodes, output_vectors, frozen_nodes,
                   edges_indices, edges_thickness, frozen_nodes)
    env.reset()

    action = {}
    action['which_node'] = np.array([0, 1])
    action['end'] = 0
    action['edge_thickness'] = np.array([x])
    action['new_node'] = np.array([[0, 2]])
    nodes_pos, edges_indices, edges_thickness, adj = env.extract_node_edge_info()
    env.step(action)
    nodes_pos, edges_indices, edges_thickness, adj = env.extract_node_edge_info()
    efficiency = env.calculate_simulation(mode='force')

    return efficiency


def objective(trial):
    x = trial.suggest_uniform("x", 0.000, 1)
    ret = f(x)
    return ret


def implement_bays():
    """ベイズ最適化を実施する関数"""

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # 探索後の最良値
    print(study.best_value)  # 432.0175613887767
    print(study.best_params)  # {'x': -3.3480816839313774}


def ddpg():
    """DDPGを利用して強化学習を行う.
    Actorの指定できる幅は0.1-1となっている"""

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    max_episodes = 500
    memory_capacity = 1e6  # バッファの容量
    gamma = 0.99  # 割引率
    tau = 1e-3  # ターゲットの更新率
    epsilon = 1.0  # ノイズの量をいじりたい場合、多分いらない
    batch_size = 64
    lr_actor = 1e-4
    lr_critic = 1e-3
    logger_interval = 10
    weight_decay = 1e-2

    test_name = "ddpg_500_v2"  # 実験名
    log_dir = "confirm/step1/ddpg_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
    os.makedirs(log_dir)

    node_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes = easy_dev()
    env = Step1Gym(node_pos, input_nodes, input_vectors,
                   output_nodes, output_vectors, frozen_nodes,
                   edges_indices, edges_thickness, frozen_nodes)
    env.reset()
    num_state = 2
    num_action = 1
    max_steps = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    actorNet = ActorNetwork(num_state, num_action).to(device)
    criticNet = CriticNetwork(num_state, num_action).to(device)
    optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(
        criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)
    replay_buffer = ReplayBuffer(capacity=memory_capacity)
    agent = DDPG(actorNet, criticNet, optimizer_actor, optimizer_critic,
                 replay_buffer, device, gamma, tau, epsilon, batch_size)

    for episode in range(max_episodes):
        observation = env.reset()
        observation = np.array([0, 1])
        total_reward = 0

        for step in range(max_steps):
            edges_thickness = agent.get_action(observation)
            action = {}
            action['which_node'] = np.array([0, 1])
            action['end'] = 0
            action['edge_thickness'] = edges_thickness
            action['new_node'] = np.array([[0, 2]])

            next_observation, _, done, _ = env.step(action)
            reward = env.calculate_simulation(mode='force')
            next_observation = np.array([0, 1])
            total_reward += reward
            agent.add_memory(observation, edges_thickness,
                             next_observation, reward, done)
            agent.train()
            observation = next_observation
            if done:
                break

        history['epoch'].append(episode+1)
        history['result_efficiency'].append(reward)
        if reward < 0:
            print(edges_thickness)

        if episode % logger_interval == 0:
            print("episode:{} total reward:{}".format(episode, total_reward))

    for episode in range(3):
        observation = env.reset()
        observation = np.array([0, 1])
        for step in range(max_steps):
            edges_thickness = agent.get_action(observation, greedy=True)
            action = {}
            action['which_node'] = np.array([0, 1])
            action['end'] = 0
            action['edge_thickness'] = edges_thickness
            action['new_node'] = np.array([[0, 2]])

            next_observation, reward, done, _ = env.step(action)
            reward = env.calculate_simulation(mode='force')
            observation = np.array([0, 1])

            if done:
                break

    env.close()
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))


def actor_critic():
    """Actor-Criticを行う．学習はDL（GCN以外）で
    Actorの指定できる幅は0.1-1となっている"""

    max_episodes = 5000
    test_name = "5000"  # 実験名

    history = {}
    history['epoch'] = []
    history['result_efficiency'] = []

    log_dir = "confirm/step1/ac_results/{}".format(test_name)
    assert not os.path.exists(log_dir), "already folder exists"
    os.makedirs(log_dir)

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

        history['epoch'].append(episode+1)
        history['result_efficiency'].append(reward)

        if episode % 10 == 0:
            print("episode:{} total reward:{}".format(episode, reward))

    env.close()
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))
