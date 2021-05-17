from check_fem_by_ansys.run import compare_apdl_barfem
import numpy as np
import os
from tqdm import tqdm
from check_fem_by_ansys.preprocess import make_random_fem_condition, make_random_fem_condition_with_ER
from env.gym_barfem import BarFemGym

max_node_num = 100
max_edge_pos = 0.3

error_num = 0

for i in tqdm(range(10)):
    nodes_pos, edges_indices, edges_thickness,\
        input_nodes, input_vectors, frozen_nodes = \
        make_random_fem_condition_with_ER(max_node_num, max_edge_pos)
    output_nodes = [0]
    output_vectors = np.array([[1, 1]])

    result = compare_apdl_barfem(nodes_pos, edges_indices, edges_thickness,
                                 input_nodes, input_vectors, frozen_nodes)

    if not result:
        error_num += 1
        log_dir = "check_fem_by_ansys/conditions/{}".format(error_num)
        os.makedirs(log_dir, exist_ok=True)
        np.save(os.path.join(log_dir, 'nodes_pos.npy'), nodes_pos)
        np.save(os.path.join(log_dir, 'edges_indices.npy'), edges_indices)
        np.save(os.path.join(log_dir, 'edges_thickness.npy'), edges_thickness)
        np.save(os.path.join(log_dir, 'input_nodes.npy'), input_nodes)
        np.save(os.path.join(log_dir, 'input_vectors.npy'), input_vectors)
        np.save(os.path.join(log_dir, 'frozen_nodes.npy'), frozen_nodes)
        env = BarFemGym(nodes_pos, input_nodes, input_vectors,
                        output_nodes, output_vectors, frozen_nodes,
                        edges_indices, edges_thickness, frozen_nodes)
        env.reset()
        env.render(save_path=os.path.join(log_dir, "image.png", display_number=True))
        break
