import numpy as np
import os
from tools.graph import calc_volume, conprocess_seperate_edge_indice_procedure, calc_efficiency, render_graph
from tools.lattice_preprocess import make_main_node_edge_info
from GA.condition import condition
from FEM.bar_fem import barfem

np_save_path = "GA/体積率に関する調査/交差していないが，値が6とおかしい構造"

nodes_pos = np.load(os.path.join(np_save_path, "nodes_pos.npy"))

edges_indices = np.load(os.path.join(np_save_path, "edges_indices.npy"))
edges_thickness = np.load(os.path.join(np_save_path, "edges_thickness.npy"))

edges_thickness = edges_thickness * 3
volume = calc_volume(nodes_pos, edges_indices, edges_thickness)
print(volume)

condition_nodes_pos, input_nodes, input_vectors, output_nodes, \
    output_vectors, frozen_nodes, condition_edges_indices, condition_edges_thickness\
    = make_main_node_edge_info(*condition(), condition_edge_thickness=0.05)  # スレンダー比を考慮し，長さ方向に対して1/20の値の幅にした


input_nodes, output_nodes, frozen_nodes, edges_thickness \
    = conprocess_seperate_edge_indice_procedure(input_nodes, output_nodes, frozen_nodes, condition_nodes_pos,
                                                condition_edges_indices, condition_edges_thickness,
                                                nodes_pos, edges_indices, edges_thickness)

displacement = barfem(nodes_pos, edges_indices, edges_thickness, input_nodes,
                      input_vectors, frozen_nodes, mode='displacement')

efficiency = calc_efficiency(input_nodes, input_vectors, output_nodes, output_vectors, displacement)

print(efficiency)

#render_graph(nodes_pos, edges_indices, edges_thickness, os.path.join("GA/体積率に関する調査", "image.png"), display_number=False)
