import optuna
from .step1_gym import Step1Gym
from .condition import easy_dev
import numpy as np


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


def main():
    """ベイズ最適化を実施する関数"""

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # 探索後の最良値
    print(study.best_value)  # 432.0175613887767
    print(study.best_params)  # {'x': -3.3480816839313774}
