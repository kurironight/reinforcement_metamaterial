"""
Use the Annealing method with FEM gym
"""

import numpy as np
from env.gym_fem import FEMGym
from tqdm import tqdm
import os
import pickle
from tools.plot import plot_efficiency_history

# PARAMETER
initial_temperature = 0.06
final_temperature = 0.001
steps = 101  # 試行回数
EDGE_THICKNESS = 0.2  # エッジの太さ
test_name = "test"

# 学習の推移
history = {}
history['epoch'] = []
history['result_efficiency'] = []

# directoryの作成
log_dir = "prior_thesis_results/{}".format(test_name)

assert not os.path.exists(log_dir), "already folder exists"
os.makedirs(log_dir)

assert steps > 100, 'steps must be bigger than 100'
# tempリストの準備
temperatures = np.linspace(
    initial_temperature, final_temperature, num=100)
temperatures = list(np.concatenate(
    [temperatures, np.ones(steps-100)*final_temperature]))


# 初期のノードの状態を抽出
origin_nodes_positions = np.array([
    [0., 0.86603], [0.5, 0.], [1., 0.86603], [1.5, 0.],
    [2., 0.86603], [2.5, 0.], [3., 0.86603], [3.5, 0.],
    [4., 0.86603], [4.5, 0.], [5., 0.86603], [5.5, 0.],
    [6., 0.86603], [6.5, 0.], [7., 0.86603], [7.5, 0.],
    [8., 0.86603], [0., 2.59808], [0.5, 1.73205], [1., 2.59808],
    [1.5, 1.73205], [2., 2.59808], [2.5, 1.73205], [3., 2.59808],
    [3.5, 1.73205], [4., 2.59808], [4.5, 1.73205], [5., 2.59808],
    [5.5, 1.73205], [6., 2.59808], [6.5, 1.73205], [7., 2.59808],
    [7.5, 1.73205], [8., 2.59808], [0., 4.33013], [0.5, 3.4641],
    [1., 4.33013], [1.5, 3.4641], [2., 4.33013], [2.5, 3.4641],
    [3., 4.33013], [3.5, 3.4641], [4., 4.33013], [4.5, 3.4641],
    [5., 4.33013], [5.5, 3.4641], [6., 4.33013], [6.5, 3.4641],
    [7., 4.33013], [7.5, 3.4641], [8., 4.33013], [0., 6.06218],
    [0.5, 5.19615], [1., 6.06218], [1.5, 5.19615], [2., 6.06218],
    [2.5, 5.19615], [3., 6.06218], [3.5, 5.19615], [4., 6.06218],
    [4.5, 5.19615], [5., 6.06218], [5.5, 5.19615], [6., 6.06218],
    [6.5, 5.19615], [7., 6.06218], [7.5, 5.19615], [8., 6.06218],
    [0., 8], [0.5, 6.9282], [1., 8], [1.5, 6.9282],
    [2., 8], [2.5, 6.9282], [3., 8], [3.5, 6.9282],
    [4., 8], [4.5, 6.9282], [5., 8], [5.5, 6.9282],
    [6., 8], [6.5, 6.9282], [7., 8], [7.5, 6.9282],
    [8., 8]])

origin_nodes_positions = origin_nodes_positions/8

origin_edges_indices = np.array([
    [0,  1], [0,  2], [0, 18], [1,  2], [1,  3], [2, 18],
    [2,  3], [2,  4], [2, 20], [3,  5], [3,  4], [4,  5],
    [4, 22], [4, 20], [4,  6], [5,  7], [5,  6], [6, 22],
    [6,  7], [6, 24], [6,  8], [7,  9], [7,  8], [8, 10],
    [8, 24], [8,  9], [8, 26], [9, 10], [9, 11], [10, 26],
    [10, 11], [10, 12], [10, 28], [11, 12], [11, 13], [12, 28],
    [12, 13], [12, 14], [12, 30], [13, 15], [13, 14], [14, 15],
    [14, 32], [14, 30], [14, 16], [15, 16], [16, 32], [17, 19],
    [17, 18], [17, 35], [18, 19], [18, 20], [19, 35], [19, 21],
    [19, 20], [19, 37], [20, 22], [20, 21], [21, 22], [21, 39],
    [21, 37], [21, 23], [22, 24], [22, 23], [23, 24], [23, 39],
    [23, 41], [23, 25], [24, 26], [24, 25], [25, 27], [25, 26],
    [25, 41], [25, 43], [26, 27], [26, 28], [27, 43], [27, 29],
    [27, 28], [27, 45], [28, 29], [28, 30], [29, 45], [29, 31],
    [29, 30], [29, 47], [30, 32], [30, 31], [31, 32], [31, 49],
    [31, 47], [31, 33], [32, 33], [33, 49], [34, 35], [34, 36],
    [34, 52], [35, 36], [35, 37], [36, 52], [36, 37], [36, 38],
    [36, 54], [37, 39], [37, 38], [38, 39], [38, 56], [38, 54],
    [38, 40], [39, 41], [39, 40], [40, 56], [40, 41], [40, 58],
    [40, 42], [41, 43], [41, 42], [42, 44], [42, 58], [42, 43],
    [42, 60], [43, 44], [43, 45], [44, 60], [44, 45], [44, 46],
    [44, 62], [45, 46], [45, 47], [46, 62], [46, 47], [46, 48],
    [46, 64], [47, 49], [47, 48], [48, 49], [48, 66], [48, 64],
    [48, 50], [49, 50], [50, 66], [51, 53], [51, 52], [51, 69],
    [52, 53], [52, 54], [53, 69], [53, 55], [53, 54], [53, 71],
    [54, 56], [54, 55], [55, 56], [55, 73], [55, 71], [55, 57],
    [56, 58], [56, 57], [57, 58], [57, 73], [57, 75], [57, 59],
    [58, 60], [58, 59], [59, 61], [59, 60], [59, 75], [59, 77],
    [60, 61], [60, 62], [61, 77], [61, 63], [61, 62], [61, 79],
    [62, 63], [62, 64], [63, 79], [63, 65], [63, 64], [63, 81],
    [64, 66], [64, 65], [65, 66], [65, 83], [65, 81], [65, 67],
    [66, 67], [67, 83], [68, 69], [68, 70], [69, 70], [69, 71],
    [70, 71], [70, 72], [71, 73], [71, 72], [72, 73], [72, 74],
    [73, 75], [73, 74], [74, 75], [74, 76], [75, 77], [75, 76],
    [76, 78], [76, 77], [77, 78], [77, 79], [78, 79], [78, 80],
    [79, 80], [79, 81], [80, 81], [80, 82], [81, 83], [81, 82],
    [82, 83], [82, 84], [83, 84],
])

edge_num = origin_edges_indices.shape[0]
origin_edges_thickness = np.ones(edge_num)*EDGE_THICKNESS

origin_input_nodes = [81, 82, 83, 84]

origin_output_nodes = [68, 69, 70, 71]

origin_frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]

condition_nodes = origin_input_nodes+origin_output_nodes+origin_frozen_nodes


env = FEMGym(origin_nodes_positions,
             origin_edges_indices, origin_edges_thickness)
env.reset()
env.render(os.path.join(log_dir, 'render_image/first.png'))

# 初期状態を作成
best_efficiency = -1000

if env.confirm_graph_is_connected():
    current_efficiency = env.calculate_simulation()
else:
    current_efficiency = -1

current_edges_indices = origin_edges_indices.copy()

for epoch, temperature in enumerate(tqdm(temperatures)):
    # 条件ノードの間にあるエッジ以外のエッジを選択
    while(1):
        chosen_edge_indice = np.random.randint(0, edge_num)
        target_edge_indice = origin_edges_indices[chosen_edge_indice]
        if not np.any(np.isin(target_edge_indice, condition_nodes)):
            break

    # proposed_efficiencyを求める
    mask = np.isin(current_edges_indices[:, 0], target_edge_indice) & np.isin(
        current_edges_indices[:, 1], target_edge_indice)
    if np.any(mask):
        # エッジが既に存在するとき
        proposed_edges_indices = current_edges_indices[~mask]
    else:
        # エッジが存在しないとき
        proposed_edges_indices = np.concatenate(
            [current_edges_indices, np.array([target_edge_indice])])

    proposed_edges_thickness = np.ones(
        proposed_edges_indices.shape[0])*EDGE_THICKNESS

    env = FEMGym(origin_nodes_positions,
                 proposed_edges_indices, proposed_edges_thickness)
    env.reset()
    if env.confirm_graph_is_connected():
        proposed_efficiency = env.calculate_simulation()
    else:
        proposed_efficiency = -1

    delta_efficiency = proposed_efficiency - current_efficiency

    if delta_efficiency < 0:
        if temperature > 0:
            prob_accept = np.exp(delta_efficiency / temperature)
        elif temperature == 0:
            prob_accept = 0
        else:
            raise RuntimeError(
                f"Cannot work with negative temperature {temperature}"
            )
        if np.random.uniform() <= 1 - prob_accept:
            # now we are rejecting the configuration
            accepted = False
        else:
            # accepting config
            accepted = True
    # efficiency has increased or not changed, we always accept
    # we only need to check if its the best one ever or not
    else:
        accepted = True

    if accepted:
        current_edges_indices = proposed_edges_indices
        current_edges_thickness = proposed_edges_thickness
        current_efficiency = proposed_efficiency

    if best_efficiency < current_efficiency:
        best_epoch = epoch
        best_efficiency = current_efficiency
        env.render(os.path.join(
            log_dir, 'render_image/{}.png'.format(epoch+1)))

    history['epoch'].append(epoch+1)
    history['result_efficiency'].append(current_efficiency)
    # 学習履歴を保存
    with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
        f.writelines('epoch %d,  result_efficiency: %.5f\n' %
                     (epoch + 1, current_efficiency))
    with open(os.path.join(log_dir, "represent_value.txt"), mode='w') as f:
        f.writelines('epoch %d,  best_efficiency: %.5f\n' %
                     (best_epoch+1, best_efficiency))
    plot_efficiency_history(history, os.path.join(
        log_dir, 'learning_effi_curve.png'))
