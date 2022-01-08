import numpy as np
from FEM.bar_fem import barfem


def easy_dev():

    nodes_pos = np.array([[1, 1],
                          [1, 0],
                          [0, 0],
                          [0, 1]])

    edges_indices = np.array([[0, 1],
                              [1, 2],
                              [2, 3]])

    edges_thickness = np.array([1.0, 1.0, 1.0])

    input_nodes = [0]
    input_vectors = np.array([[1, 0]])
    frozen_nodes = [2]
    output_nodes = [3]
    output_vectors = np.array([[1, 0]])

    return nodes_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes
