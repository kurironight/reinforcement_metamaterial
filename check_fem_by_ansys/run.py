from ansys.mapdl.core import launch_mapdl
from FEM.bar_fem import barfem

import numpy as np

max_node_num = 250
max_edge_num = 250


def compare_apdl_barfem(nodes_pos, edges_indices, edges_thickness,
                        input_nodes, input_vectors, frozen_nodes, tmax=1000, eps=1.0e-7):

    # C言語を用いたbarfem
    displacement = barfem(nodes_pos, edges_indices, edges_thickness, input_nodes,
                          input_vectors, frozen_nodes, mode="force", tmax=tmax, eps=eps)

    # APDLの設定
    mapdl = launch_mapdl()

    mapdl.finish()
    mapdl.clear()
    mapdl.prep7()

    # 材料物性値の設定
    mapdl.et(1, 3)
    mapdl.mp("ex", 1, 1)  # ヤング率
    mapdl.mp("prxy", 1, 0.3)  # ポアソン比
    mapdl.mat(1)

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        mapdl.n(i + 1, node_pos[0], node_pos[1], 0)

    # エッジ部分の設定
    for i, edges_indice in enumerate(edges_indices):
        b = 0.2  # 奥行
        h = edges_thickness[i]
        mapdl.r(i + 1, b * h, b * (h * h**2) / 12, h, 0)  # A,I,height=y方向の長さ,SHEARZ
        mapdl.real(i + 1)
        mapdl.e(edges_indice[0] + 1, edges_indice[1] + 1)

    mapdl.finish()
    mapdl.run('/solu')
    mapdl.antype('static')

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        mapdl.d(i + 1, "all", 0)
    # 外力設定
    for i, input_vector in enumerate(input_vectors):
        mapdl.f(input_nodes[i] + 1, "FX", input_vector[0])
        mapdl.f(input_nodes[i] + 1, "FY", input_vector[1])
    # 解析開始
    mapdl.solve()
    mapdl.finish()

    # 結果出力
    x_disp = mapdl.post_processing.nodal_displacement('X')
    y_disp = mapdl.post_processing.nodal_displacement('Y')
    z_rot = mapdl.post_processing.nodal_rotation('Z')

    ansys_disp = np.stack([x_disp, y_disp, z_rot]).T.flatten()

    mapdl.exit()  # ポート番号などをリセットするために必要

    # 厳密には少し小さい値の部分が異なる為，allclose構文を利用
    return np.allclose(ansys_disp, displacement)
