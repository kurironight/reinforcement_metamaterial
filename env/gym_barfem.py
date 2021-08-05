from .gym_metamech import MetamechGym
from FEM.bar_fem import barfem
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tools.graph import remove_node_which_nontouchable_in_edge_indices, calc_efficiency, render_graph


class BarFemGym(MetamechGym):
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes):
        super(BarFemGym, self).__init__(node_pos, input_nodes, input_vectors,
                                        output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes)
        assert len(self.output_nodes) == 1, "output_node should be 1 size of list"
        assert self.output_vectors.shape[0] == 1 and self.output_vectors.shape[1] == 2, "output_vector should be [1,2]"

    def calculate_simulation(self, mode='displacement'):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()
        node_num = nodes_pos.shape[0]
        assert node_num >= np.max(
            edges_indices), 'edges_indicesに，ノード数以上のindexを示しているものが発生'
        input_nodes, output_nodes, frozen_nodes, nodes_pos, edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, nodes_pos, edges_indices)
        displacement = barfem(nodes_pos, edges_indices, edges_thickness, input_nodes,
                              self.input_vectors, frozen_nodes, mode)
        efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)
        return efficiency

    # 環境の描画
    def render(self, save_path="image/image.png", display_number=False):
        """グラフを図示

        Args:
            save_path (str, optional): 図を保存するパス. Defaults to "image/image.png".
            display_number (bool, optional): ノードに番号をつけるか付けないか. Defaults to False.
        """
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        render_graph(nodes_pos, edges_indices, edges_thickness, save_path, display_number)
