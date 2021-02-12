from .gym_metamech import MetamechGym, MAX_EDGE_THICKNESS
from FEM.bar_fem import barfem
import numpy as np


class BarFemGym(MetamechGym):
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness):
        super(BarFemGym, self).__init__(node_pos, input_nodes, input_vectors,
                                        output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness)
        assert len(self.output_nodes) == 1, "output_node should be 1 size of list"
        assert self.output_vectors.shape[0] == 1 and self.output_vectors.shape[1] == 2, "output_vector should be [1,2]"

    def calculate_simulation(self):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()
        displacement = barfem(nodes_pos, edges_indices, edges_thickness, self.input_nodes,
                              self.input_vectors, self.frozen_nodes)

        efficiency = np.dot(self.output_vectors, displacement[[
                            self.output_nodes[0]*3+0, self.output_nodes[0]*3+1]])
        return efficiency
