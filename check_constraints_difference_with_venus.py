import numpy as np
import os
from tools.graph import *
from GA.condition import *
from GA.GA_class import *


nodes_pos, input_nodes, input_vectors, output_nodes, output_vectors,\
    frozen_nodes, edges_indices, edges_thickness, L, A\
    = venus_trap_condition(0.2, 0.025)

GA_type = VenusFrytrap_GA
problem = GA_type(2, 2)
new_edges_indices = np.concatenate([edges_indices, np.array([[10, 11]])])
add_nodes_pos = np.array([[0.2, 0.2], [0.8, 0.8]])
x_max = np.max(nodes_pos[:, 0])
for i, node_pos in enumerate(add_nodes_pos):
    add_nodes_pos[i][0] = add_nodes_pos[i][0] * x_max
    add_nodes_pos[i][1] = problem.convert_ratio_y_coord_to_y_coord(add_nodes_pos[i][0], add_nodes_pos[i][1])
new_nodes_pos = np.concatenate([nodes_pos, add_nodes_pos])
new_edges_thickness = np.concatenate([edges_thickness, [0.025]])
render_graph(new_nodes_pos, new_edges_indices, new_edges_thickness, "venus_image.png", display_number=True)


condition_edge_thickness = 0.01
nodes_pos, input_nodes, input_vectors, output_nodes, \
    output_vectors, frozen_nodes, edges_indices, edges_thickness\
    = make_main_node_edge_info(*condition(), condition_edge_thickness=condition_edge_thickness)  # スレンダー比を考慮し，長さ方向に対して1/20の値の幅にした
new_edges_indices = np.concatenate([edges_indices, np.array([[10, 11]])])
add_nodes_pos = np.array([[0.2, 0.2], [0.8, 0.8]])
new_nodes_pos = np.concatenate([nodes_pos, add_nodes_pos])
new_edges_thickness = np.concatenate([edges_thickness, [0.01]])
render_graph(new_nodes_pos, new_edges_indices, new_edges_thickness, "easy_image.png", display_number=True)
