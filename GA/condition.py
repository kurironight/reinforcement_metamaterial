import numpy as np
from tools.graph import *


def condition():
    # 初期のノードの状態を抽出
    origin_nodes_positions = np.array([
        [0., 0.86603], [0.5, 0.], [1., 0.86603], [1.5, 0.],
        [2., 0.86603], [2.5, 0.], [3., 0.86603], [3.5, 0.],
        [4., 0.86603], [4.5, 0.], [5., 0.86603], [5.5, 0.],
        [6., 0.86603], [6.5, 0.], [7., 0.86603], [7.5, 0.],
        [8., 0.86603], [0., 2.59808], [0.5, 1.73205], [1., 2.59808],
        [1.5, 1.73205], [2., 2.59808], [2.5, 1.73205], [3., 2.59808],
        [3.5, 1.73205], [4., 2.59808], [4.5, 1.73205], [5., 2.59808],
        [5.5, 1.73205], [6., 2.59808], [6.5, 1.73205], [7., 2.59808],
        [7.5, 1.73205], [8., 2.59808], [0., 4.33013], [0.5, 3.4641],
        [1., 4.33013], [1.5, 3.4641], [2., 4.33013], [2.5, 3.4641],
        [3., 4.33013], [3.5, 3.4641], [4., 4.33013], [4.5, 3.4641],
        [5., 4.33013], [5.5, 3.4641], [6., 4.33013], [6.5, 3.4641],
        [7., 4.33013], [7.5, 3.4641], [8., 4.33013], [0., 6.06218],
        [0.5, 5.19615], [1., 6.06218], [1.5, 5.19615], [2., 6.06218],
        [2.5, 5.19615], [3., 6.06218], [3.5, 5.19615], [4., 6.06218],
        [4.5, 5.19615], [5., 6.06218], [5.5, 5.19615], [6., 6.06218],
        [6.5, 5.19615], [7., 6.06218], [7.5, 5.19615], [8., 6.06218],
        [0., 8], [0.5, 6.9282], [1., 8], [1.5, 6.9282],
        [2., 8], [2.5, 6.9282], [3., 8], [3.5, 6.9282],
        [4., 8], [4.5, 6.9282], [5., 8], [5.5, 6.9282],
        [6., 8], [6.5, 6.9282], [7., 8], [7.5, 6.9282],
        [8., 8]])

    origin_nodes_positions = origin_nodes_positions / 8

    origin_edges_indices = np.array([
        [0, 1], [0, 2], [0, 18], [1, 2], [1, 3], [2, 18],
        [2, 3], [2, 4], [2, 20], [3, 5], [3, 4], [4, 5],
        [4, 22], [4, 20], [4, 6], [5, 7], [5, 6], [6, 22],
        [6, 7], [6, 24], [6, 8], [7, 9], [7, 8], [8, 10],
        [8, 24], [8, 9], [8, 26], [9, 10], [9, 11], [10, 26],
        [10, 11], [10, 12], [10, 28], [11, 12], [11, 13], [12, 28],
        [12, 13], [12, 14], [12, 30], [13, 15], [13, 14], [14, 15],
        [14, 32], [14, 30], [14, 16], [15, 16], [16, 32], [17, 19],
        [17, 18], [17, 35], [18, 19], [18, 20], [19, 35], [19, 21],
        [19, 20], [19, 37], [20, 22], [20, 21], [21, 22], [21, 39],
        [21, 37], [21, 23], [22, 24], [22, 23], [23, 24], [23, 39],
        [23, 41], [23, 25], [24, 26], [24, 25], [25, 27], [25, 26],
        [25, 41], [25, 43], [26, 27], [26, 28], [27, 43], [27, 29],
        [27, 28], [27, 45], [28, 29], [28, 30], [29, 45], [29, 31],
        [29, 30], [29, 47], [30, 32], [30, 31], [31, 32], [31, 49],
        [31, 47], [31, 33], [32, 33], [33, 49], [34, 35], [34, 36],
        [34, 52], [35, 36], [35, 37], [36, 52], [36, 37], [36, 38],
        [36, 54], [37, 39], [37, 38], [38, 39], [38, 56], [38, 54],
        [38, 40], [39, 41], [39, 40], [40, 56], [40, 41], [40, 58],
        [40, 42], [41, 43], [41, 42], [42, 44], [42, 58], [42, 43],
        [42, 60], [43, 44], [43, 45], [44, 60], [44, 45], [44, 46],
        [44, 62], [45, 46], [45, 47], [46, 62], [46, 47], [46, 48],
        [46, 64], [47, 49], [47, 48], [48, 49], [48, 66], [48, 64],
        [48, 50], [49, 50], [50, 66], [51, 53], [51, 52], [51, 69],
        [52, 53], [52, 54], [53, 69], [53, 55], [53, 54], [53, 71],
        [54, 56], [54, 55], [55, 56], [55, 73], [55, 71], [55, 57],
        [56, 58], [56, 57], [57, 58], [57, 73], [57, 75], [57, 59],
        [58, 60], [58, 59], [59, 61], [59, 60], [59, 75], [59, 77],
        [60, 61], [60, 62], [61, 77], [61, 63], [61, 62], [61, 79],
        [62, 63], [62, 64], [63, 79], [63, 65], [63, 64], [63, 81],
        [64, 66], [64, 65], [65, 66], [65, 83], [65, 81], [65, 67],
        [66, 67], [67, 83], [68, 69], [68, 70], [69, 70], [69, 71],
        [70, 71], [70, 72], [71, 73], [71, 72], [72, 73], [72, 74],
        [73, 75], [73, 74], [74, 75], [74, 76], [75, 77], [75, 76],
        [76, 78], [76, 77], [77, 78], [77, 79], [78, 79], [78, 80],
        [79, 80], [79, 81], [80, 81], [80, 82], [81, 83], [81, 82],
        [82, 83], [82, 84], [83, 84],
    ])

    # origin_edges_indices = np.concatenate(
    #    [origin_edges_indices, [[81, 68], [68, 9]]])
    origin_input_nodes = [84]
    origin_input_vectors = np.array([
        [0., -1]
    ])

    origin_output_nodes = [68]
    origin_output_vectors = np.array([
        [-1, 0],
    ])

    origin_frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]

    return origin_nodes_positions, origin_edges_indices, origin_input_nodes,\
        origin_input_vectors, origin_output_nodes, origin_output_vectors, origin_frozen_nodes


def condition_only_input_output():
    origin_nodes_positions, origin_edges_indices, origin_input_nodes,\
        origin_input_vectors, origin_output_nodes, origin_output_vectors, _ = condition()

    return origin_nodes_positions, origin_edges_indices, origin_input_nodes,\
        origin_input_vectors, origin_output_nodes, origin_output_vectors, []


def venus_trap_condition(b,edge_width):
    midrib_length = 6  # 長さ6mm
    left_thickness = 1  # 左端の幅1mm
    right_thickness = 0.5  # 左端の幅0.5mm

    edge_width = 0.025  # 条件エッジの太さ

    record_point_1 = np.array([234, 111], dtype=float)
    record_point_2 = np.array([304, 177], dtype=float)
    record_point_3 = np.array([374, 210], dtype=float)
    record_point_4 = np.array([444, 221], dtype=float)

    record_vector1 = record_point_2 - record_point_1
    record_vector2 = record_point_3 - record_point_2
    record_vector3 = record_point_4 - record_point_3

    record_length = np.linalg.norm(record_vector1) + np.linalg.norm(record_vector2) + np.linalg.norm(record_vector3)

    ratio = record_length / midrib_length  # 修正倍率

    point1 = record_point_1 / ratio
    point2 = record_point_2 / ratio
    point3 = record_point_3 / ratio
    point4 = record_point_4 / ratio

    vector1 = record_vector1 / np.linalg.norm(record_vector1)

    vertical_vector1 = np.array([point2[1] - point1[1], point1[0] - point2[0]])
    vertical_vector2 = np.array([point3[1] - point2[1], point2[0] - point3[0]])
    vertical_vector3 = np.array([point4[1] - point3[1], point3[0] - point4[0]])

    vertical_vector1 = vertical_vector1 / np.linalg.norm(vertical_vector1)
    vertical_vector2 = vertical_vector2 / np.linalg.norm(vertical_vector2)
    vertical_vector3 = vertical_vector3 / np.linalg.norm(vertical_vector3)

    d2 = np.linalg.norm(point3 - point2)
    d3 = np.linalg.norm(point4 - point3)

    point5 = point1 - left_thickness * vertical_vector1
    point6 = point2 - (right_thickness * (1 + ((d2 + d3) / midrib_length))) * vertical_vector1
    point7 = point2 - (right_thickness * (1 + ((d2 + d3) / midrib_length))) * vertical_vector2
    point8 = point3 - (right_thickness * (1 + ((d3) / midrib_length))) * vertical_vector2
    point9 = point3 - (right_thickness * (1 + ((d3) / midrib_length))) * vertical_vector3
    point10 = point4 - (right_thickness) * vertical_vector3

    point0 = point1 - (point1[0] - point5[0]) / vector1[0] * vector1

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8,
                          point9,
                          point10]) - point0
    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [5, 6],
                             [6, 7],
                             [7, 8],
                             [8, 9],
                             [9, 10],
                             [0, 5],
                             [4, 10]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    # 点6,7や点8.9を統一化し，簡潔に表現
    point6 = (point6 + point7) / 2
    point7 = (point8 + point9) / 2
    point8 = point10

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8]) - point0

    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [5, 6],
                             [6, 7],
                             [7, 8],
                             [0, 5],
                             [4, 8]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    # 0,5が固定点となる為，5,6間のエッジにかかる圧力を表すのに十分なノード数とは呼べない．よって，5,6の間にノードを追加．
    point9 = point8
    point8 = point7
    point7 = point6
    point6 = ((point6 - point5)[0] / (point1[0])) * (point6 - point5) + point5

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8,
                          point9]) - point0

    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [5, 6],
                             [6, 7],
                             [7, 8],
                             [8, 9],
                             [0, 5],
                             [4, 9]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])

    # 力を計算
    p = 1  # 圧力の大きさ

    def calc_vertical_vector(point1, point2):
        vertical_vector = np.array([point2[1] - point1[1], point1[0] - point2[0]])
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
        return vertical_vector

    def calc_downer_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 下方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice1 = edges_indices[edges_indices[:, 1] == node][0]
        edge_indice2 = edges_indices[edges_indices[:, 0] == node][0]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (edge1_length / 2 * vertical_vector1 + edge2_length / 2 * vertical_vector2) * p

    def calc_upper_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 上方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice = edges_indices[(edges_indices[:, 1] == node) | (edges_indices[:, 0] == node)]
        edge_indice1 = edge_indice[0]
        edge_indice2 = edge_indice[1]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (-edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    node1_pressure_vector = calc_downer_pressure(1, nodes_pos, lengths, edges_indices, p)
    node2_pressure_vector = calc_downer_pressure(2, nodes_pos, lengths, edges_indices, p)
    node3_pressure_vector = calc_downer_pressure(3, nodes_pos, lengths, edges_indices, p)
    node4_pressure_vector = calc_downer_pressure(4, nodes_pos, lengths, edges_indices, p)
    # print(node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector)

    node6_pressure_vector = calc_upper_pressure(6, nodes_pos, lengths, edges_indices, p)
    node7_pressure_vector = calc_upper_pressure(7, nodes_pos, lengths, edges_indices, p)
    node8_pressure_vector = calc_upper_pressure(8, nodes_pos, lengths, edges_indices, p)

    # [4,9]に関してのみ例外対応
    vertical_vector1 = calc_vertical_vector(nodes_pos[4], nodes_pos[9])
    edge1_length = lengths[find_edge_indice_index([4, 9], edges_indices)]
    vertical_vector2 = calc_vertical_vector(nodes_pos[8], nodes_pos[9])

    edge2_length = lengths[find_edge_indice_index([8, 9], edges_indices)]
    node9_pressure_vector = (edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    # print(node6_pressure_vector, node7_pressure_vector, node8_pressure_vector, node9_pressure_vector)
    input_nodes = [1, 2, 3, 4, 6, 7, 8, 9]
    input_vectors = np.array([node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector,
                              node6_pressure_vector, node7_pressure_vector, node8_pressure_vector, node9_pressure_vector])
    output_nodes = [9]
    output_vectors = calc_vertical_vector(nodes_pos[9], nodes_pos[0])
    frozen_nodes = [0, 5]

    # 代表長さ，代表面積の測定
    L = np.sum(lengths)  # 代表長さ
    A = b * L  # 代表面積

    return nodes_pos, input_nodes, input_vectors, output_nodes, output_vectors,\
        frozen_nodes, edges_indices, edges_thickness, L, A


def venus_trap_condition_high_resolution(b):
    midrib_length = 6  # 長さ6mm
    left_thickness = 1  # 左端の幅1mm
    right_thickness = 0.5  # 左端の幅0.5mm

    edge_width = 0.025  # 条件エッジの太さ

    record_point_1 = np.array([234, 111], dtype=float)
    record_point_2 = np.array([304, 177], dtype=float)
    record_point_3 = np.array([374, 210], dtype=float)
    record_point_4 = np.array([444, 221], dtype=float)

    record_vector1 = record_point_2 - record_point_1
    record_vector2 = record_point_3 - record_point_2
    record_vector3 = record_point_4 - record_point_3

    record_length = np.linalg.norm(record_vector1) + np.linalg.norm(record_vector2) + np.linalg.norm(record_vector3)

    ratio = record_length / midrib_length  # 修正倍率

    point1 = record_point_1 / ratio
    point2 = record_point_2 / ratio
    point3 = record_point_3 / ratio
    point4 = record_point_4 / ratio

    vector1 = record_vector1 / np.linalg.norm(record_vector1)

    vertical_vector1 = np.array([point2[1] - point1[1], point1[0] - point2[0]])
    vertical_vector2 = np.array([point3[1] - point2[1], point2[0] - point3[0]])
    vertical_vector3 = np.array([point4[1] - point3[1], point3[0] - point4[0]])

    vertical_vector1 = vertical_vector1 / np.linalg.norm(vertical_vector1)
    vertical_vector2 = vertical_vector2 / np.linalg.norm(vertical_vector2)
    vertical_vector3 = vertical_vector3 / np.linalg.norm(vertical_vector3)

    d2 = np.linalg.norm(point3 - point2)
    d3 = np.linalg.norm(point4 - point3)

    point5 = point1 - left_thickness * vertical_vector1
    point6 = point2 - (right_thickness * (1 + ((d2 + d3) / midrib_length))) * vertical_vector1
    point7 = point2 - (right_thickness * (1 + ((d2 + d3) / midrib_length))) * vertical_vector2
    point8 = point3 - (right_thickness * (1 + ((d3) / midrib_length))) * vertical_vector2
    point9 = point3 - (right_thickness * (1 + ((d3) / midrib_length))) * vertical_vector3
    point10 = point4 - (right_thickness) * vertical_vector3

    point0 = point1 - (point1[0] - point5[0]) / vector1[0] * vector1

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8,
                          point9,
                          point10]) - point0
    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [5, 6],
                             [6, 7],
                             [7, 8],
                             [8, 9],
                             [9, 10],
                             [0, 5],
                             [4, 10]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    # 点6,7や点8.9を統一化し，簡潔に表現
    point6 = (point6 + point7) / 2
    point7 = (point8 + point9) / 2
    point8 = point10

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8]) - point0

    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [5, 6],
                             [6, 7],
                             [7, 8],
                             [0, 5],
                             [4, 8]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    # 0,5が固定点となる為，5,6間のエッジにかかる圧力を表すのに十分なノード数とは呼べない．よって，5,6の間にノードを追加．
    point9 = point8
    point8 = point7
    point7 = point6
    point6 = ((point6 - point5)[0] / (point1[0])) * (point6 - point5) + point5

    point12 = (point3 + point4) / 2
    point13 = (point8 + point9) / 2

    nodes_pos = np.array([point0,
                          point1,
                          point2,
                          point3,
                          point12,
                          point4,
                          point5,
                          point6,
                          point7,
                          point8,
                          point13,
                          point9]) - point0

    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [4, 5],
                             [6, 7],
                             [7, 8],
                             [8, 9],
                             [9, 10],
                             [10, 11],
                             [5, 11],
                             [0, 6]])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])

    # 力を計算
    p = 1  # 圧力の大きさ

    def calc_vertical_vector(point1, point2):
        vertical_vector = np.array([point2[1] - point1[1], point1[0] - point2[0]])
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
        return vertical_vector

    def calc_downer_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 下方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice1 = edges_indices[edges_indices[:, 1] == node][0]
        edge_indice2 = edges_indices[edges_indices[:, 0] == node][0]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (edge1_length / 2 * vertical_vector1 + edge2_length / 2 * vertical_vector2) * p

    def calc_upper_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 上方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice = edges_indices[(edges_indices[:, 1] == node) | (edges_indices[:, 0] == node)]
        edge_indice1 = edge_indice[0]
        edge_indice2 = edge_indice[1]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (-edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    node1_pressure_vector = calc_downer_pressure(1, nodes_pos, lengths, edges_indices, p)
    node2_pressure_vector = calc_downer_pressure(2, nodes_pos, lengths, edges_indices, p)
    node3_pressure_vector = calc_downer_pressure(3, nodes_pos, lengths, edges_indices, p)
    node4_pressure_vector = calc_downer_pressure(4, nodes_pos, lengths, edges_indices, p)
    node5_pressure_vector = calc_downer_pressure(5, nodes_pos, lengths, edges_indices, p)

    node7_pressure_vector = calc_upper_pressure(7, nodes_pos, lengths, edges_indices, p)
    node8_pressure_vector = calc_upper_pressure(8, nodes_pos, lengths, edges_indices, p)
    node9_pressure_vector = calc_upper_pressure(9, nodes_pos, lengths, edges_indices, p)
    node10_pressure_vector = calc_upper_pressure(10, nodes_pos, lengths, edges_indices, p)

    # [4,9]に関してのみ例外対応
    vertical_vector1 = calc_vertical_vector(nodes_pos[5], nodes_pos[11])
    edge1_length = lengths[find_edge_indice_index([5, 11], edges_indices)]
    vertical_vector2 = calc_vertical_vector(nodes_pos[10], nodes_pos[11])

    edge2_length = lengths[find_edge_indice_index([10, 11], edges_indices)]
    node11_pressure_vector = (edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    input_nodes = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    input_vectors = np.array([node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector, node5_pressure_vector,
                              node7_pressure_vector, node8_pressure_vector, node9_pressure_vector, node10_pressure_vector, node11_pressure_vector])
    output_nodes = [11]
    output_vectors = calc_vertical_vector(nodes_pos[11], nodes_pos[0])
    frozen_nodes = [0, 6]

    # 代表長さ，代表面積の測定
    L = np.sum(lengths)  # 代表長さ
    A = b * L  # 代表面積

    return nodes_pos, input_nodes, input_vectors, output_nodes, output_vectors,\
        frozen_nodes, edges_indices, edges_thickness, L, A


def venus_trap_condition_high_resolution_v2(b):
    midrib_length = 6  # 長さ6mm
    left_thickness = 1  # 左端の幅1mm
    right_thickness = 0.5  # 左端の幅0.5mm

    edge_width = 0.025  # 条件エッジの太さ

    nodes_pos, input_nodes, input_vectors, output_nodes, output_vectors,\
        frozen_nodes, edges_indices, edges_thickness, L, A = venus_trap_condition_high_resolution(b)
    ref_nodes_pos = nodes_pos.copy()
    nodes_pos = np.array([ref_nodes_pos[0],
                          ref_nodes_pos[1],
                          ref_nodes_pos[2],
                          (ref_nodes_pos[2] + ref_nodes_pos[3]) / 2,
                          ref_nodes_pos[3],
                          ref_nodes_pos[4],
                          ref_nodes_pos[5],
                          ref_nodes_pos[6],
                          ref_nodes_pos[7],
                          ref_nodes_pos[8],
                          (ref_nodes_pos[8] + ref_nodes_pos[9]) / 2,
                          ref_nodes_pos[9],
                          ref_nodes_pos[10],
                          ref_nodes_pos[11]])

    edges_indices = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4],
                             [4, 5],
                             [5, 6],
                             [7, 8],
                             [8, 9],
                             [9, 10],
                             [10, 11],
                             [11, 12],
                             [12, 13],
                             [0, 7],
                             [6, 13], ])

    edges_thickness = np.ones(edges_indices.shape[0]) * edge_width

    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])

    # 力を計算
    p = 1  # 圧力の大きさ

    def calc_vertical_vector(point1, point2):
        vertical_vector = np.array([point2[1] - point1[1], point1[0] - point2[0]])
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
        return vertical_vector

    def calc_downer_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 下方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice1 = edges_indices[edges_indices[:, 1] == node][0]
        edge_indice2 = edges_indices[edges_indices[:, 0] == node][0]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (edge1_length / 2 * vertical_vector1 + edge2_length / 2 * vertical_vector2) * p

    def calc_upper_pressure(node, nodes_pos, lengths, edges_indices, p):
        # 上方向の圧力を測定する為の関数．
        # nodeには，求めたいノードの番号を入れる
        edge_indice = edges_indices[(edges_indices[:, 1] == node) | (edges_indices[:, 0] == node)]
        edge_indice1 = edge_indice[0]
        edge_indice2 = edge_indice[1]
        vertical_vector1 = calc_vertical_vector(nodes_pos[edge_indice1[0]], nodes_pos[edge_indice1[1]])
        edge1_length = lengths[find_edge_indice_index(edge_indice1, edges_indices)]
        vertical_vector2 = calc_vertical_vector(nodes_pos[edge_indice2[0]], nodes_pos[edge_indice2[1]])
        edge2_length = lengths[find_edge_indice_index(edge_indice2, edges_indices)]
        return (-edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    node1_pressure_vector = calc_downer_pressure(1, nodes_pos, lengths, edges_indices, p)
    node2_pressure_vector = calc_downer_pressure(2, nodes_pos, lengths, edges_indices, p)
    node3_pressure_vector = calc_downer_pressure(3, nodes_pos, lengths, edges_indices, p)
    node4_pressure_vector = calc_downer_pressure(4, nodes_pos, lengths, edges_indices, p)
    node5_pressure_vector = calc_downer_pressure(5, nodes_pos, lengths, edges_indices, p)
    node6_pressure_vector = calc_downer_pressure(6, nodes_pos, lengths, edges_indices, p)

    node8_pressure_vector = calc_upper_pressure(8, nodes_pos, lengths, edges_indices, p)
    node9_pressure_vector = calc_upper_pressure(9, nodes_pos, lengths, edges_indices, p)
    node10_pressure_vector = calc_upper_pressure(10, nodes_pos, lengths, edges_indices, p)
    node11_pressure_vector = calc_upper_pressure(11, nodes_pos, lengths, edges_indices, p)
    node12_pressure_vector = calc_upper_pressure(12, nodes_pos, lengths, edges_indices, p)

    # [4,9]に関してのみ例外対応
    vertical_vector1 = calc_vertical_vector(nodes_pos[6], nodes_pos[13])
    edge1_length = lengths[find_edge_indice_index([6, 13], edges_indices)]
    vertical_vector2 = calc_vertical_vector(nodes_pos[12], nodes_pos[13])
    edge2_length = lengths[find_edge_indice_index([12, 13], edges_indices)]
    node13_pressure_vector = (edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

    input_nodes = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    input_vectors = np.array([node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector, node5_pressure_vector,
                             node6_pressure_vector, node8_pressure_vector, node9_pressure_vector, node10_pressure_vector, node11_pressure_vector,
                             node12_pressure_vector, node13_pressure_vector])
    output_nodes = [13]
    output_vectors = calc_vertical_vector(nodes_pos[13], nodes_pos[0])
    frozen_nodes = [0, 7]

    # 代表長さ，代表面積の測定
    L = np.sum(lengths)  # 代表長さ
    A = b * L  # 代表面積

    return nodes_pos, input_nodes, input_vectors, output_nodes, output_vectors,\
        frozen_nodes, edges_indices, edges_thickness, L, A
