import numpy as np
from tools.graph import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from FEM.bar_fem import barfem, barfem_anti

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
vector2 = record_vector2 / np.linalg.norm(record_vector2)
vector3 = record_vector3 / np.linalg.norm(record_vector3)

vertical_vector1 = np.array([point2[1] - point1[1], point1[0] - point2[0]])
vertical_vector2 = np.array([point3[1] - point2[1], point2[0] - point3[0]])
vertical_vector3 = np.array([point4[1] - point3[1], point3[0] - point4[0]])

vertical_vector1 = vertical_vector1 / np.linalg.norm(vertical_vector1)
vertical_vector2 = vertical_vector2 / np.linalg.norm(vertical_vector2)
vertical_vector3 = vertical_vector3 / np.linalg.norm(vertical_vector3)

d1 = np.linalg.norm(point2 - point1)
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

# nemoto_vector = (point10) - point0
# print(90 - np.degrees(np.arctan2(nemoto_vector[1], nemoto_vector[0])))
render_graph(nodes_pos, edges_indices, edges_thickness, "近似直線.png", display_number=True)

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

render_graph(nodes_pos, edges_indices, edges_thickness, "近似直線_簡潔化.png", display_number=True)

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

render_graph(nodes_pos, edges_indices, edges_thickness, "近似直線_圧力考慮.png", display_number=True)
edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])
# print(lengths)
#render_pixel_graph(nodes_pos / np.max(nodes_pos), edges_indices, edges_thickness / np.max(nodes_pos), "近似直線_圧力考慮_ピクセル.png", 1000)

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
#print(node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector)

node6_pressure_vector = calc_upper_pressure(6, nodes_pos, lengths, edges_indices, p)
node7_pressure_vector = calc_upper_pressure(7, nodes_pos, lengths, edges_indices, p)
node8_pressure_vector = calc_upper_pressure(8, nodes_pos, lengths, edges_indices, p)

# [4,9]に関してのみ例外対応
vertical_vector1 = calc_vertical_vector(nodes_pos[4], nodes_pos[9])
edge1_length = lengths[find_edge_indice_index([4, 9], edges_indices)]
vertical_vector2 = calc_vertical_vector(nodes_pos[8], nodes_pos[9])

edge2_length = lengths[find_edge_indice_index([8, 9], edges_indices)]
node9_pressure_vector = (edge1_length / 2 * vertical_vector1 + -edge2_length / 2 * vertical_vector2) * p

#print(node6_pressure_vector, node7_pressure_vector, node8_pressure_vector, node9_pressure_vector)
input_nodes = [1, 2, 3, 4, 6, 7, 8, 9]
input_vectors = np.array([node1_pressure_vector, node2_pressure_vector, node3_pressure_vector, node4_pressure_vector,
                          node6_pressure_vector, node7_pressure_vector, node8_pressure_vector, node9_pressure_vector])
frozen_nodes = [0, 5]
#print(np.dot(node1_pressure_vector, point2 - point0))

# 力の方向を図示
marker_size = 40  # 図示するときのノードのサイズ
character_size = 20  # ノードの文字のサイズ
edge_size = 100
starts = nodes_pos[edges_indices[:, 0]]
ends = nodes_pos[edges_indices[:, 1]]
lines = [(start, end) for start, end in zip(starts, ends)]
lines = LineCollection(lines, linewidths=edges_thickness * edge_size)
plt.clf()  # Matplotlib内の図全体をクリアする
#plt.figure(figsize=(6, 6))
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(lines)
ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], s=marker_size, c="red", zorder=2)
for i, txt in enumerate(["{}".format(i) for i in range(nodes_pos.shape[0])]):
    ax.annotate(txt, (nodes_pos[i, 0], nodes_pos[i, 1]), size=character_size, horizontalalignment="center", verticalalignment="center")

for node, pressure in zip(input_nodes, input_vectors):
    ax.quiver(nodes_pos[node, 0], nodes_pos[node, 1], pressure[0],
              pressure[1], color="red",
              angles='xy', scale_units='xy', scale=2)

ax.set_xlim(-0.1, 7)
ax.set_ylim(-0.1, 7)

plt.savefig("圧力ベクトル")
plt.close()

# 実際の計測
displacement, stresses = barfem_anti(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, mode="force")
print(displacement[[9 * 3 + 0, 9 * 3 + 1]] / 100000000)

# 代表長さ，代表面積の測定
b = 0.2  # 奥行
L = np.sum(lengths)  # 代表長さ
A = b * L  # 代表面積
print(L, A)
