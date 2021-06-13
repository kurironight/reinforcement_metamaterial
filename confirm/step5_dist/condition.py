import numpy as np


def easy_dev():

    nodes_pos = np.array([[0, 0],
                          [0.5, 0],
                          [0.5, 0.5],
                          [0, 0.5]])
    nodes_pos[:, 0] += 0.25
    nodes_pos[:, 1] += 0.25

    edges_indices = np.array([[0, 1],
                              [1, 2],
                              [0, 3],
                              [2, 3]])

    edges_thickness = np.array([1, 1, 1, 1])

    input_nodes = [2]
    input_vectors = np.array([[1, 0]])
    frozen_nodes = [1]
    output_nodes = [0]
    output_vectors = np.array([[0.5, 0.5]])

    return nodes_pos, input_nodes, input_vectors,\
        output_nodes, output_vectors, frozen_nodes,\
        edges_indices, edges_thickness, frozen_nodes
