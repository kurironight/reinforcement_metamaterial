import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history(history, save_path):
    epochs = history['epoch']
    loss = history['loss']
    epochs = np.array(epochs)
    loss = np.array(loss)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, loss, label='loss')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_title("learning curve")
    plt.savefig(save_path)
    plt.close()


def plot_reward_history(history, save_path):
    epochs = history['epoch']
    ep_reward = history['ep_reward']
    epochs = np.array(epochs)
    ep_reward = np.array(ep_reward)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, ep_reward, label='ep_reward')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.legend()
    ax.set_title("reward curve")
    plt.savefig(save_path)
    plt.close()


def plot_efficiency_history(history, save_path):
    epochs = history['epoch']
    result_efficiency = history['result_efficiency']
    epochs = np.array(epochs)
    result_efficiency = np.array(result_efficiency)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, result_efficiency, label='efficiency')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.legend()
    ax.set_title("efficiency curve")
    plt.savefig(save_path)
    plt.close()


def plot_steps_history(history, save_path):
    epochs = history['epoch']
    results = history['steps']
    epochs = np.array(epochs)
    results = np.array(results)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, results, label='steps')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.legend()
    ax.set_title("steps curve")
    plt.savefig(save_path)
    plt.close()
