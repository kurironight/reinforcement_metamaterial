import numpy as np
import os
from env.gym_barfem import BarFemGym
from GA.condition import condition
from tools.lattice_preprocess import make_main_node_edge_info
from tools.graph import *


np_save_path = "GA/result/confirm"

nodes_pos = np.load(os.path.join(np_save_path, "nodes_pos.npy"))
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

print(efficiency)
