import numpy as np
from .graph import *


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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
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

    edges_indices, edges_thickness = make_same_slope_group_edge(nodes_pos, edges_indices, edges_thickness)
    assert np.allclose(edges_indices, np.array([[1, 2],
                                                [1, 3],
                                                [2, 5],
                                                [3, 4]]) - 1)
    assert np.allclose(edges_thickness, np.array([3, 1, 2, 2]))


def test_seperate_y_axis_slope_group():
    nodes_pos = np.array([[0, 1],
                          [1, 0],
                          [0, 3],
                          [0, 2],
                          [4, 0]])

    edges_indices = np.array([[1, 3],
                              [2, 3],
                              [3, 4],
                              [4, 5]]) - 1
    same_slope_group = separate_same_slope_group(nodes_pos, edges_indices)
    assert np.array_equal(same_slope_group, np.array([[[1, 3], [3, 4]]]) - 1)
