from ansys.mapdl.core import launch_mapdl
from FEM.bar_fem import barfem
from check_fem_by_ansys.run import compare_apdl_barfem
import numpy as np
import os
from tqdm import tqdm
from check_fem_by_ansys.preprocess import make_random_fem_condition
from env.gym_barfem import BarFemGym

max_node_num = 30
max_edge_num = 30


for i in tqdm(range(100)):
    nodes_pos, edges_indices, edges_thickness,\
        input_nodes, input_vectors, frozen_nodes = \
        make_random_fem_condition(max_node_num, max_edge_num)
    output_nodes = [0]
    output_vectors = np.array([[1, 1]])
    env = BarFemGym(nodes_pos, input_nodes, input_vectors,
                    output_nodes, output_vectors, frozen_nodes,
                    edges_indices, edges_thickness, frozen_nodes)
    env.reset()
    env.render(save_path="check_fem_by_ansys/conditions/{}.png".format(i))

    result = compare_apdl_barfem(nodes_pos, edges_indices, edges_thickness,
                                 input_nodes, input_vectors, frozen_nodes, log_dir="check_fem_by_ansys/conditions/{}".format(i))
    if not result:
        print("失敗")
        break
