from .gym_metamech import MetamechGym, MAX_EDGE_THICKNESS
from FEM.bar_fem import barfem
import numpy as np


class BarFemGym(MetamechGym):
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes):
        super(BarFemGym, self).__init__(node_pos, input_nodes, input_vectors,
                                        output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness, condition_nodes)
        assert len(self.output_nodes) == 1, "output_node should be 1 size of list"
        assert self.output_vectors.shape[0] == 1 and self.output_vectors.shape[1] == 2, "output_vector should be [1,2]"

    def calculate_simulation(self, mode='displacement'):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()
        input_nodes = np.array(self.input_nodes)
        frozen_nodes = np.array(self.frozen_nodes)
        node_num = nodes_pos.shape[0]
        assert node_num >= np.max(
            edges_indices), 'edges_indicesに，ノード数以上のindexを示しているものが発生'
        mask = np.isin(np.arange(node_num), edges_indices)
        if not np.all(mask):  # barfemの為，edge_indicesではnodes_posの内，触れられていないノードが存在しないように処理する
            processed_input_nodes = input_nodes.copy()
            processed_frozen_nodes = frozen_nodes.copy()
            processed_edges_indices = edges_indices.copy()
            prior_index = np.arange(node_num)[mask]
            processed_nodes_pos = nodes_pos[mask]
            for index, prior_index in enumerate(prior_index):
                if index != prior_index:
                    processed_edges_indices[edges_indices ==
                                            prior_index] = index
                    # input_nodesとfrozen_nodes部分のラベルを変更
                    processed_input_nodes[input_nodes == prior_index] = index
                    processed_frozen_nodes[frozen_nodes == prior_index] = index
            nodes_pos = processed_nodes_pos
            edges_indices = processed_edges_indices
            input_nodes = processed_input_nodes
            frozen_nodes = processed_frozen_nodes
        input_nodes = input_nodes.tolist()
        frozen_nodes = frozen_nodes.tolist()
        displacement = barfem(nodes_pos, edges_indices, edges_thickness, input_nodes,
                              self.input_vectors, frozen_nodes, mode)

        efficiency = np.dot(self.output_vectors, displacement[[
                            self.output_nodes[0] * 3 + 0, self.output_nodes[0] * 3 + 1]])
        return efficiency
