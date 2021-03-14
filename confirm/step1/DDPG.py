import gym
import torch.optim as optim
import copy
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import deque, namedtuple
import random
from itertools import islice

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.memory = deque([], maxlen=int(capacity))

    def append(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return islice(self.memory,  self.__len__()-batch_size, None)
        # return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def length(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


class ActorNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_action)

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = 0.1+0.9*torch.sigmoid(self.fc3(h))
        return y


class CriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-4):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size+num_action, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, action):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(torch.cat([h, action], dim=1)))
        y = self.fc3(h)
        return y


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.num_steps = 0

        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
        return sigma

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.num_steps += 1

        return x


class DDPG:
    def __init__(self, actor, critic, optimizer_actor, optimizer_critic, replay_buffer, device, gamma=0.99, tau=1e-3, epsilon=1.0, batch_size=64):
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.random_process = OrnsteinUhlenbeckProcess(
            size=actor.num_action)

        self.num_state = actor.num_state
        self.num_action = actor.num_action

    def add_memory(self, *args):
        self.replay_buffer.append(*args)

    def reset_memory(self):
        self.replay_buffer.reset()

    def get_action(self, state, greedy=False):
        state_tensor = torch.tensor(
            state, dtype=torch.float, device=self.device).view(-1, self.num_state)
        action = self.actor(state_tensor)
        if not greedy:
            action += self.epsilon * \
                torch.tensor(self.random_process.sample(),
                             dtype=torch.float, device=self.device)
        action = torch.clamp(action, min=0.1, max=1)

        return action.squeeze(0).detach().cpu().numpy()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(
            batch.state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(
            batch.next_state, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float)
        not_done = np.array([(not done) for done in batch.done])
        not_done_batch = torch.tensor(
            not_done, device=self.device, dtype=torch.float).unsqueeze(1)

        # need to change
        qvalue = self.critic(state_batch, action_batch)
        next_qvalue = self.critic_target(
            next_state_batch, self.actor_target(next_state_batch))
        target_qvalue = reward_batch + \
            (self.gamma * next_qvalue * not_done_batch)
        critic_loss = F.mse_loss(qvalue, target_qvalue)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # soft parameter update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)


"""
max_episodes = 10
memory_capacity = 1e6  # バッファの容量
gamma = 0.99  # 割引率
tau = 1e-3  # ターゲットの更新率
epsilon = 1.0  # ノイズの量をいじりたい場合、多分いらない
batch_size = 64
lr_actor = 1e-4
lr_critic = 1e-3
logger_interval = 10
weight_decay = 1e-2

env = gym.make('Pendulum-v0')
num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = env.spec.max_episode_steps

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
    total_reward = 0

    for step in range(max_steps):
        action = agent.get_action(observation)

        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_memory(observation, action, next_observation, reward, done)

        agent.train()

        observation = next_observation

        if done:
            break

    if episode % logger_interval == 0:
        print("episode:{} total reward:{}".format(episode, total_reward))

for episode in range(3):
    observation = env.reset()
    for step in range(max_steps):
        action = agent.get_action(observation, greedy=True)
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation

        if done:
            break

env.close()
"""
