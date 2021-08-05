import numpy as np
import os
from env.gym_barfem import BarFemGym
from GA.condition import condition
from tools.lattice_preprocess import make_main_node_edge_info
from tools.graph import *


search_edge_indice_1 = [1, 11]  # 探したいエッジの幅のindice
search_edge_indice_2 = [0, 2]  # 探したいエッジの幅のindice


np_save_path = "GA/confirm_data_for_cross_process/50"

nodes_pos = np.load(os.path.join(np_save_path, "nodes_pos.npy"))
#nodes_pos[10, :] = np.array([0.5, 0.5])
edges_indices = np.load(os.path.join(np_save_path, "edges_indices.npy"))
edges_thickness = np.load(os.path.join(np_save_path, "edges_thickness.npy"))

condition_nodes_pos, input_nodes, input_vectors, output_nodes, \
    output_vectors, frozen_nodes, condition_edges_indices, condition_edges_thickness\
    = make_main_node_edge_info(*condition(), condition_edge_thickness=0.05)  # スレンダー比を考慮し，長さ方向に対して1/20の値の幅にした

env = BarFemGym(nodes_pos, input_nodes, input_vectors,
                output_nodes, output_vectors, frozen_nodes,
                edges_indices, edges_thickness, frozen_nodes)
env.reset()
efficiency = env.calculate_simulation()
#env.render(save_path="image/image_preprocess.png", display_number=True)

# 同じノード位置にあるものを排除する．
processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
# 傾きが一致するものをグループ分けし，エッジ分割を行う．
processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

processed_nodes_pos, processed_edges_indices, processed_edges_thickness =\
    seperate_cross_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

# 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
    preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

input_nodes, output_nodes, frozen_nodes, processed_edges_thickness \
    = conprocess_seperate_edge_indice_procedure(input_nodes, output_nodes, frozen_nodes, condition_nodes_pos, condition_edges_indices, condition_edges_thickness,
                                                processed_nodes_pos, processed_edges_indices, processed_edges_thickness)


env = BarFemGym(processed_nodes_pos, input_nodes, input_vectors,
                output_nodes, output_vectors, frozen_nodes,
                processed_edges_indices, processed_edges_thickness, frozen_nodes)
env.reset()
efficiency = env.calculate_simulation()
env.render(save_path="image/image_proprocess.png", display_number=True)
print(efficiency)
