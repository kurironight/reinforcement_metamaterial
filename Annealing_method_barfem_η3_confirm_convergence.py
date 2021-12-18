"""
アニーリング法で得た結果が本当に収束しているのかを確認する関数
エッジの候補219こ全てを選択して性能が上がらない場合，デッドロックとなる為，収束したといえる．
"""

import numpy as np
from env.gym_barfem import BarFemOutputGym
from tqdm import tqdm
import os

trial_time = 1  # アニーリング法を何回行うか
EDGE_THICKNESS = 0.01  # エッジの太さ
check_dir = "Annealing_results_η3_test"
for i in range(trial_time):
    result_dir = os.path.join(check_dir, "{}/graph_info".format(i))

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

    origin_nodes_positions = origin_nodes_positions / 8

    origin_edges_indices = np.array([
        [0, 1], [0, 2], [0, 18], [1, 2], [1, 3], [2, 18],
        [2, 3], [2, 4], [2, 20], [3, 5], [3, 4], [4, 5],
        [4, 22], [4, 20], [4, 6], [5, 7], [5, 6], [6, 22],
        [6, 7], [6, 24], [6, 8], [7, 9], [7, 8], [8, 10],
        [8, 24], [8, 9], [8, 26], [9, 10], [9, 11], [10, 26],
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
    origin_edges_thickness = np.ones(edge_num) * EDGE_THICKNESS

    origin_frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]

    origin_input_vectors = np.array([
        [0., -1],
    ])
    origin_output_vectors = np.array([
        [-1, 0],
    ])
    barfem_input_nodes = [84]
    barfem_output_nodes = [68]
    condition_nodes = barfem_input_nodes + barfem_output_nodes + origin_frozen_nodes

    result_dir = "001/{}/graph_info".format(i)

    current_edges_indices = np.load(os.path.join(result_dir, 'edges_indices.npy'))
    current_edges_thickness = np.load(os.path.join(result_dir, 'edges_thickness.npy'))

    env = BarFemOutputGym(origin_nodes_positions, barfem_input_nodes, origin_input_vectors,
                          barfem_output_nodes, origin_output_vectors, origin_frozen_nodes,
                          current_edges_indices, current_edges_thickness, origin_frozen_nodes)
    env.reset()
    current_efficiency = env.calculate_simulation()

    # for chosen_edge_index in tqdm(range(edge_num)):
    for chosen_edge_index in tqdm(range(1)):
        # 条件ノードの間にあるエッジ以外のエッジを選択
        chosen_edge_index = 134
        target_edge_indice = origin_edges_indices[chosen_edge_index]
        if np.isin(target_edge_indice[0], condition_nodes) & np.isin(target_edge_indice[1], condition_nodes):
            continue

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
            proposed_edges_indices.shape[0]) * EDGE_THICKNESS

        env = BarFemOutputGym(origin_nodes_positions, barfem_input_nodes, origin_input_vectors,
                              barfem_output_nodes, origin_output_vectors, origin_frozen_nodes,
                              proposed_edges_indices, proposed_edges_thickness, origin_frozen_nodes)
        env.reset()
        if env.confirm_graph_is_connected():
            proposed_efficiency = env.calculate_simulation()
        else:
            proposed_efficiency = -1000000000000
        delta_efficiency = proposed_efficiency - current_efficiency
        if (delta_efficiency > 0) & (delta_efficiency > current_efficiency * 10**(-10)):
            print(chosen_edge_index)
