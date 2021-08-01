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

    sl_groups, independent_group = separate_same_line_group(nodes_pos, edges_indices)

    revised_edges_indices = np.empty((0, 2), int)
    revised_edges_thickness = np.empty(0)

    if sl_groups != []:
        for sl_group_edge_indices in sl_groups:
            sl_group_edge_indices = np.array(sl_group_edge_indices).reshape((-1, 2))
            sl_group_edges_thickness = np.array([edges_thickness[find_edge_indice_index(target_edges_indice, edges_indices)] for target_edges_indice in sl_group_edge_indices])
            revised_sl_group_edge_indices, revised_sl_edges_thickness = \
                make_same_line_group_edge(nodes_pos, sl_group_edge_indices, sl_group_edges_thickness)
            revised_edges_indices = np.append(revised_edges_indices, revised_sl_group_edge_indices, axis=0)
            revised_edges_thickness = np.append(revised_edges_thickness, revised_sl_edges_thickness, axis=0)

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

    sl_groups, independent_group = separate_same_line_group(nodes_pos, edges_indices)

    revised_edges_indices = np.empty((0, 2), int)
    revised_edges_thickness = np.empty(0)

    if sl_groups != []:
        for sl_group_edge_indices in sl_groups:
            sl_group_edge_indices = np.array(sl_group_edge_indices).reshape((-1, 2))
            sl_group_edges_thickness = np.array([edges_thickness[find_edge_indice_index(target_edges_indice, edges_indices)] for target_edges_indice in sl_group_edge_indices])
            revised_sl_group_edge_indices, revised_sl_edges_thickness = \
                make_same_line_group_edge(nodes_pos, sl_group_edge_indices, sl_group_edges_thickness)
            revised_edges_indices = np.append(revised_edges_indices, revised_sl_group_edge_indices, axis=0)
            revised_edges_thickness = np.append(revised_edges_thickness, revised_sl_edges_thickness, axis=0)

    if independent_group != []:
        id_group_edges_thickness = np.array([edges_thickness[find_edge_indice_index(target_edges_indice, edges_indices)] for target_edges_indice in independent_group])
        revised_edges_indices = np.append(revised_edges_indices, independent_group, axis=0)
        revised_edges_thickness = np.append(revised_edges_thickness, id_group_edges_thickness, axis=0)

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
