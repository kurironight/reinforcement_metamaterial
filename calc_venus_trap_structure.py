import numpy as np
from tools.graph import render_graph

midrib_length = 6  # 長さ6mm
left_thickness = 1  # 左端の幅1mm
right_thickness = 0.5  # 左端の幅0.5mm


record_point_1 = np.array([234, 111], dtype=float)
record_point_2 = np.array([304, 177], dtype=float)
record_point_3 = np.array([374, 210], dtype=float)
record_point_4 = np.array([444, 221], dtype=float)

record_vector1 = record_point_2 - record_point_1
record_vector2 = record_point_3 - record_point_2
record_vector3 = record_point_4 - record_point_3

record_length = np.linalg.norm(record_vector1) + np.linalg.norm(record_vector2) + np.linalg.norm(record_vector3)
print(record_length)

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

edges_thickness = np.ones(edges_indices.shape[0]) * 0.01

nemoto_vector = point4 - point0
print(np.degrees(np.arctan2(nemoto_vector[1], nemoto_vector[0])))
render_graph(nodes_pos, edges_indices, edges_thickness, "image.png", display_number=True)
