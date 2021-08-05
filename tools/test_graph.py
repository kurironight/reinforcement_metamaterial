import numpy as np
from .graph import *
from env.gym_barfem import BarFemGym


def test_1():
    nodes_pos = np.array([[1, 0],
                          [4, 0],
                          [2, 0],
                          [5, 0],
                          [0, 0],
                          [3, 0]])

    edges_indices = np.array([[0, 1],
                              [2, 3],
                              [4, 5]])

    edges_thickness = np.array([6, 3, 2])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 3],
                                                [1, 5],
                                                [2, 4],
                                                [2, 6],
                                                [3, 6]]) - 1)
    assert np.allclose(edges_thickness, np.array([6, 2, 3, 6, 6]))


def test_2():
    nodes_pos = np.array([[1, 0],
                          [4, 0],
                          [2, 0],
                          [5, 0],
                          [0, 0],
                          [3, 0]])

    edges_indices = np.array([[0, 1],
                              [2, 3],
                              [4, 5]])

    edges_thickness = np.array([2, 6, 4])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 3],
                                                [1, 5],
                                                [2, 4],
                                                [2, 6],
                                                [3, 6]]) - 1)
    assert np.allclose(edges_thickness, np.array([4, 4, 6, 6, 6]))


def test_3():
    nodes_pos = np.array([[1, 0],
                          [4, 0],
                          [2, 0],
                          [5, 0],
                          [0, 0],
                          [3, 0]])

    edges_indices = np.array([[0, 1],
                              [2, 3],
                              [4, 5]])

    edges_thickness = np.array([4, 2, 6])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 3],
                                                [1, 5],
                                                [2, 4],
                                                [2, 6],
                                                [3, 6]]) - 1)
    assert np.allclose(edges_thickness, np.array([6, 6, 2, 4, 6]))


def test_4():
    nodes_pos = np.array([[2, 0],
                          [3, 0],
                          [4, 0],
                          [1, 0],
                          [5, 0]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [4, 5]]) - 1

    edges_thickness = np.array([2, 3, 1])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 2],
                                                [1, 4],
                                                [2, 3],
                                                [3, 5]]) - 1)
    assert np.allclose(edges_thickness, np.array([2, 1, 3, 1]))


def test_5():
    nodes_pos = np.array([[1, 0],
                          [3, 0],
                          [2, 0],
                          [0, 0],
                          [4, 0]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [4, 5]]) - 1

    edges_thickness = np.array([4, 6, 2])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 3],
                                                [1, 4],
                                                [2, 3],
                                                [2, 5]]) - 1)
    assert np.allclose(edges_thickness, np.array([4, 2, 6, 2]))


def test_6():
    nodes_pos = np.array([[1, 0],
                          [2, 0],
                          [3, 0],
                          [0, 0],
                          [4, 0]])

    edges_indices = np.array([[1, 2],
                              [1, 3],
                              [4, 5]]) - 1

    edges_thickness = np.array([4, 3, 2])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 2],
                                                [1, 4],
                                                [2, 3],
                                                [3, 5]]) - 1)
    assert np.allclose(edges_thickness, np.array([4, 2, 3, 2]))


def test_8():
    nodes_pos = np.array([[2, 0],
                          [3, 0],
                          [1, 0],
                          [0, 0],
                          [4, 0]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [3, 4],
                              [4, 5],
                              [2, 5]]) - 1

    edges_thickness = np.array([3, 1, 2, 1, 2])

    edges_indices, edges_thickness = make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 2],
                                                [1, 3],
                                                [2, 5],
                                                [3, 4]]) - 1)
    assert np.allclose(edges_thickness, np.array([3, 1, 2, 2]))


def test_seperate_y_axis_line_group():
    nodes_pos = np.array([[0, 1],
                          [1, 0],
                          [0, 3],
                          [0, 2],
                          [4, 0]])

    edges_indices = np.array([[1, 3],
                              [2, 3],
                              [3, 4],
                              [4, 5]]) - 1
    same_line_group, independent_group = separate_same_line_group(nodes_pos, edges_indices)
    assert np.array_equal(same_line_group, np.array([[[1, 3], [3, 4]]]) - 1)
    assert np.array_equal(independent_group, np.array([[2, 3], [4, 5]]) - 1)


def test_seperate_line_group():
    nodes_pos = np.array([[0, 1],
                          [1, 0],
                          [0, 3],
                          [0, 2],
                          [4, 0],
                          [0, 0],
                          [1, 1],
                          [3, 3]])

    edges_indices = np.array([[1, 3],
                              [2, 3],
                              [3, 4],
                              [4, 5],
                              [6, 7],
                              [6, 8]]) - 1
    same_line_group, independent_group = separate_same_line_group(nodes_pos, edges_indices)
    assert np.array_equal(same_line_group, np.array([[[1, 3], [3, 4]],
                                                     [[6, 7], [6, 8]]]) - 1)
    assert np.array_equal(independent_group, np.array([[2, 3], [4, 5]]) - 1)


def test_seperate_same_line_procedure():
    nodes_pos = np.array([[0, 0],
                          [0.25, 0],
                          [0.5, 0],
                          [0.75, 0],
                          [1, 0],
                          [1, 1],
                          [0, 1],
                          [0.5, 0.5],
                          [0.25, 0.25]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [3, 4],
                              [4, 5],
                              [1, 6],
                              [5, 8],
                              [7, 8],
                              [8, 9],
                              [1, 3],
                              [2, 4],
                              [1, 9]]) - 1

    edges_thickness = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 2, 3, 2, 2, 5, 1])

    revised_edges_indices, revised_edges_thickness = separate_same_line_procedure(nodes_pos, edges_indices, edges_thickness)

    answer_edges_indices = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5],
                                     [1, 9],
                                     [8, 9],
                                     [6, 8],
                                     [5, 8],
                                     [7, 8]]) - 1

    answer_edges_thickness = np.array([2, 5, 5, 1, 1.5, 2, 1.5, 2, 3])

    for i, t in zip(answer_edges_indices, answer_edges_thickness):
        assert t == revised_edges_thickness[find_edge_indice_index(i, revised_edges_indices)]


def test_seperate_same_line_plus_independent_procedure():
    nodes_pos = np.array([[0, 0],
                          [0.25, 0],
                          [0.5, 0],
                          [0.75, 0],
                          [1, 0],
                          [1, 1],
                          [0, 1],
                          [0.5, 0.5],
                          [0.25, 0.25]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [3, 4],
                              [4, 5],
                              [1, 6],
                              [5, 8],
                              [7, 8],
                              [8, 9],
                              [1, 3],
                              [2, 4],
                              [1, 9],
                              [3, 8],
                              [5, 6],
                              [1, 7]]) - 1

    edges_thickness = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 2, 3, 2, 2, 5, 1, 1, 2, 3])

    revised_edges_indices, revised_edges_thickness = separate_same_line_procedure(nodes_pos, edges_indices, edges_thickness)

    answer_edges_indices = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5],
                                     [1, 9],
                                     [8, 9],
                                     [6, 8],
                                     [5, 8],
                                     [7, 8],
                                     [3, 8],
                                     [5, 6],
                                     [1, 7]]) - 1

    answer_edges_thickness = np.array([2, 5, 5, 1, 1.5, 2, 1.5, 2, 3, 1, 2, 3])

    for i, t in zip(answer_edges_indices, answer_edges_thickness):
        assert t == revised_edges_thickness[find_edge_indice_index(i, revised_edges_indices)]


def test_seperate_same_line_plus_independent_same_node_procedure():
    # ノード8と10，ノード9と11が一致する
    nodes_pos = np.array([[0, 0],
                          [0.25, 0],
                          [0.5, 0],
                          [0.75, 0],
                          [1, 0],
                          [1, 1],
                          [0, 1],
                          [0.5, 0.5],
                          [0.25, 0.25],
                          [0.5, 0.5], [0.25, 0.25], [0.25, 0.25]])

    edges_indices = np.array([[1, 2],
                              [2, 3],
                              [3, 4],
                              [4, 5],
                              [1, 6],
                              [5, 8],
                              [7, 8],
                              [8, 9],
                              [1, 3],
                              [2, 4],
                              [1, 9],
                              [3, 8],
                              [5, 6],
                              [1, 7],
                              [2, 11],
                              [4, 10],
                              [5, 10],
                              [3, 12], ]) - 1

    edges_thickness = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 2, 3, 2, 2, 5, 1, 1, 2, 3, 1, 1, 3, 2])

    processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)

    revised_edges_indices, revised_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

    revised_edge_points = np.array([[nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]] for edges_indice in revised_edges_indices])

    answer_edges_indices = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5],
                                     [1, 9],
                                     [8, 9],
                                     [6, 8],
                                     [5, 8],
                                     [7, 8],
                                     [3, 8],
                                     [5, 6],
                                     [1, 7], [2, 9], [4, 8], [3, 9]]) - 1

    answer_edge_points = np.array([[nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]] for edges_indice in answer_edges_indices])

    answer_edges_thickness = np.array([2, 5, 5, 1, 1.5, 2, 1.5, 3, 3, 1, 2, 3, 1, 1, 2])

    for i, t in zip(answer_edge_points, answer_edges_thickness):
        target_index = np.argwhere(np.array([np.array_equal(j, i[0]) for j in revised_edge_points[:, 0]]) & np.array([np.array_equal(j, i[1]) for j in revised_edge_points[:, 1]]))
        if target_index.shape[0] == 0:
            target_index = np.argwhere(np.array([np.array_equal(j, i[1]) for j in revised_edge_points[:, 0]]) & np.array([np.array_equal(j, i[0]) for j in revised_edge_points[:, 1]]))
        assert t == revised_edges_thickness[target_index]


def test_seperate_same_line_plus_independent_same_node_procedure2():
    nodes_pos = np.array([[0, 1],
                          [1, 1],
                          [1, 0],
                          [0, 0],
                          [0.5, 0.5],
                          [0.5, 0],
                          [0.5, 0.5],
                          [0.75, 0.25],
                          [0.25, 0.75],
                          [0.75, 0.25]])

    edges_indices = np.array([[1, 2],
                              [1, 9],
                              [5, 8],
                              [7, 10],
                              [8, 9],
                              [3, 8],
                              [4, 5],
                              [4, 6],
                              [3, 6]]) - 1

    edges_thickness = np.array([1.0, 3, 1, 2, 2, 1, 2, 1, 1])

    processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)

    revised_edges_indices, revised_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

    revised_edge_points = np.array([[processed_nodes_pos[edges_indice[0]], processed_nodes_pos[edges_indice[1]]] for edges_indice in revised_edges_indices])

    answer_edges_indices = np.array([[1, 2],
                                     [1, 9],
                                     [5, 9],
                                     [5, 8],
                                     [3, 8],
                                     [4, 5],
                                     [4, 6],
                                     [3, 6]]) - 1

    answer_edge_points = np.array([[nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]] for edges_indice in answer_edges_indices])

    answer_edges_thickness = np.array([1, 3, 2, 2, 1, 2, 1, 1])
    for i, t in zip(answer_edge_points, answer_edges_thickness):
        target_index = np.argwhere(np.array([np.array_equal(j, i[0]) for j in revised_edge_points[:, 0]]) & np.array([np.array_equal(j, i[1]) for j in revised_edge_points[:, 1]]))
        if target_index.shape[0] == 0:
            target_index = np.argwhere(np.array([np.array_equal(j, i[1]) for j in revised_edge_points[:, 0]]) & np.array([np.array_equal(j, i[0]) for j in revised_edge_points[:, 1]]))
        assert t == revised_edges_thickness[target_index]


def test_seperate_cross_edge():
    nodes_pos = np.array([[0, 0],
                          [0.25, 0],
                          [1, 0],
                          [1, 0.75],
                          [1, 1],
                          [0.75, 1],
                          [0.25, 1],
                          [0, 1]])

    edges_indices = np.array([[1, 5],
                              [2, 7],
                              [3, 6],
                              [4, 8]]) - 1

    edges_thickness = np.array([1.0, 3, 2, 1])

    processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)

    processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

    processed_nodes_pos, processed_edges_indices, processed_edges_thickness =\
        seperate_cross_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

    # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
    processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
        preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

    processed_edge_points = np.array([[processed_nodes_pos[edges_indice[0]], processed_nodes_pos[edges_indice[1]]] for edges_indice in processed_edges_indices])

    answer_nodes_pos = np.array([[0., 0.],
                                 [0.25, 0.],
                                 [1., 0.],
                                 [1., 0.75],
                                 [1., 1.],
                                 [0.75, 1.],
                                 [0.25, 1.],
                                 [0., 1.],
                                 [0.25, 0.25],
                                 [0.8, 0.8],
                                 [0.25, 0.9375]])

    answer_edges_indices = np.array([[1, 9],
                                     [9, 10],
                                     [5, 10],
                                     [2, 9],
                                     [9, 11],
                                     [7, 11],
                                     [3, 10],
                                     [6, 10],
                                     [4, 10],
                                     [10, 11],
                                     [8, 11]]) - 1

    answer_edge_points = np.array([[answer_nodes_pos[edges_indice[0]], answer_nodes_pos[edges_indice[1]]] for edges_indice in answer_edges_indices])

    answer_edges_thickness = np.array([1, 1, 1, 3, 3, 3, 2, 2, 1, 1, 1])

    for i, t in zip(answer_edge_points, answer_edges_thickness):
        target_index = np.argwhere(np.array([np.array_equal(j, i[0]) for j in processed_edge_points[:, 0]]) & np.array([np.array_equal(j, i[1]) for j in processed_edge_points[:, 1]]))
        if target_index.shape[0] == 0:
            target_index = np.argwhere(np.array([np.array_equal(j, i[1]) for j in processed_edge_points[:, 0]]) & np.array([np.array_equal(j, i[0]) for j in processed_edge_points[:, 1]]))
        assert t == processed_edges_thickness[target_index]
