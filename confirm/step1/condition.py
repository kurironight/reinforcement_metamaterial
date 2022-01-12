import numpy as np
from FEM.bar_fem import barfem


def easy_dev():
    L = 1
    A = 1

    nodes_pos = np.array([[0, 0],
                          [L, 0]])

    edges_indices = np.array([[0, 1]])

    edges_thickness = np.array([1.0 * A])

    input_nodes = [1]
    input_vectors = np.array([[1, 0]])
    frozen_nodes = [0]
    output_nodes = [1]
    output_vectors = np.array([[1.0, 0]])

    return nodes_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes
