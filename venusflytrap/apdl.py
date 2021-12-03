import numpy as np
from FEM.bar_fem import barfem_anti
from tools.graph import *
import pickle
import matplotlib.pyplot as plt
import os
from GA.condition import venus_trap_condition
from .make_structure import make_node


def convert_graph_info_to_mapdl_code(nodes_pos, edges_indices, edges_thickness, input_nodes,
                                     input_vectors, frozen_nodes, b, daihenkei=True, substep=1e9, highest_step=1e9, min_step=1e5):
    path = 'haetorigusa.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,3 ! sets element type 1 to beam3, the 2d beam element in ANSYS")
    f.write("\n")
    f.write("mp,ex,1,10 ! sets modulus of mtl 1")
    f.write("\n")
    f.write("mp,prxy,1,0.3")
    f.write("\n")
    f.write("mat,1")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("n,{},{},{},0".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")

    # エッジ部分の設定
    for i, edges_indice in enumerate(edges_indices):
        h = edges_thickness[i]
        f.write("r,{},{},{},{},0".format(i + 1, b * h, b * (h * h**2) / 12, h))
        f.write("\n")
        f.write("real,{}".format(i + 1))
        f.write("\n")
        f.write("e,{},{}".format(edges_indice[0] + 1, edges_indice[1] + 1))
        f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")

    f.write('finish')
    f.write("\n")
    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("d,{},all,0".format(i + 1))
        f.write("\n")
    # 外力設定
    for i, input_vector in enumerate(input_vectors):
        f.write("f,{},FX,{}".format(input_nodes[i] + 1, input_vector[0]))
        f.write("\n")
        f.write("f,{},FY,{}".format(input_nodes[i] + 1, input_vector[1]))
        f.write("\n")
    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def convert_graph_info_to_mapdl_code_pressure(nodes_pos, edges_indices, edges_thickness,
                                              frozen_nodes, b, pressure, daihenkei=True, substep=1e3, highest_step=1e3, min_step=1e1):
    path = 'haetorigusa.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,3 ! sets element type 1 to beam3, the 2d beam element in ANSYS")
    f.write("\n")
    f.write("mp,ex,1,10 ! sets modulus of mtl 1")
    f.write("\n")
    f.write("mp,prxy,1,0.3")
    f.write("\n")
    f.write("mat,1")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("n,{},{},{},0".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")

    # エッジ部分の設定
    for i, edges_indice in enumerate(edges_indices):
        h = edges_thickness[i]
        f.write("r,{},{},{},{},0".format(i + 1, b * h, b * (h * h**2) / 12, h))
        f.write("\n")
        f.write("real,{}".format(i + 1))
        f.write("\n")
        f.write("e,{},{}".format(edges_indice[0] + 1, edges_indice[1] + 1))
        f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")

    f.write('finish')
    f.write("\n")
    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("d,{},all,0".format(i + 1))
        f.write("\n")
    # 外力設定
    for i, edges_indice in enumerate(edges_indices):
        f.write("ESEL,S,ELEM,,{},{}".format(i + 1, i + 1))
        f.write("\n")
        if i <= 4:
            f.write("SFBEAM,{},1,PRES,{}".format(i + 1, pressure))
            f.write("\n")
        if (4 < i) and (i <= 8):
            f.write("SFBEAM,{},1,PRES,-{}".format(i + 1, pressure))
            f.write("\n")
    f.write("ESEL,ALL")
    f.write("\n")
    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def convert_graph_info_to_mapdl_code_pressure_inner_thick_same(nodes_pos, edges_indices, edges_thickness,
                                                               frozen_nodes, b, pressure, big_ratio, daihenkei=True, substep=1e3, highest_step=1e3, min_step=1e1):
    path = 'haetorigusa.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,3 ! sets element type 1 to beam3, the 2d beam element in ANSYS")
    f.write("\n")
    f.write("mp,ex,1,10 ! sets modulus of mtl 1")
    f.write("\n")
    f.write("mp,prxy,1,0.3")
    f.write("\n")
    f.write("mat,1")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("n,{},{},{},0".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")

    # エッジ部分の設定
    condition_edges_indices = np.array([[0, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [5, 6], [6, 7], [7, 8], [8, 9]])
    for i, edges_indice in enumerate(edges_indices):
        if np.any((edges_indice[0] == condition_edges_indices[:, 0]) & (edges_indice[1] == condition_edges_indices[:, 1])):
            h = edges_thickness[i] * big_ratio
        else:
            h = edges_thickness[i]
        f.write("r,{},{},{},{},0".format(i + 1, b * h, b * (h * h**2) / 12, h))
        f.write("\n")
        f.write("real,{}".format(i + 1))
        f.write("\n")
        f.write("e,{},{}".format(edges_indice[0] + 1, edges_indice[1] + 1))
        f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")

    f.write('finish')
    f.write("\n")
    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("d,{},all,0".format(i + 1))
        f.write("\n")
    # 外力設定
    for i, edges_indice in enumerate(edges_indices):
        f.write("ESEL,S,ELEM,,{},{}".format(i + 1, i + 1))
        f.write("\n")
        if i <= 4:
            f.write("SFBEAM,{},1,PRES,{}".format(i + 1, pressure))
            f.write("\n")
        if (4 < i) and (i <= 8):
            f.write("SFBEAM,{},1,PRES,-{}".format(i + 1, pressure))
            f.write("\n")
    f.write("ESEL,ALL")
    f.write("\n")
    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def convert_graph_info_to_mapdl_188(nodes_pos, edges_indices, edges_thickness, input_nodes,
                                    input_vectors, frozen_nodes, b, daihenkei=True, substep=1e9, highest_step=1e9, min_step=1e5):
    path = 'haetorigusa.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,188 ! sets element type 1 to beam3, the 2d beam element in ANSYS")
    f.write("\n")
    f.write("keyopt,1,3,2")
    f.write("\n")
    # f.write("keyopt,1,5,1")
    # f.write("\n")
    f.write("mp,ex,1,10 ! sets modulus of mtl 1")
    f.write("\n")
    f.write("mp,prxy,1,0.3")
    f.write("\n")
    f.write("mat,1")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("k,{},{},{}".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")

    # エッジ部分の設定
    for i, edges_indice in enumerate(edges_indices):
        h = edges_thickness[i]
        f.write("sectype,{},beam,rect,,0".format(i + 1))
        f.write("\n")
        f.write("secdata,{},{},2,2".format(h, b))  # 高さ，幅，高さに沿ったセル数，幅に沿ったセル数
        f.write("\n")

        f.write("l,{},{}".format(edges_indice[0] + 1, edges_indice[1] + 1))
        f.write("\n")
        f.write("lesize,{},,,5 !lineを10分割でmeshする".format(i + 1))
        f.write("\n")
        f.write("latt,1,1,1,,,,{} !最後の数値でsecnumを指定する".format(i + 1))
        f.write("\n")
    f.write("lmesh,ALL")
    f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")
    f.write("/eshape,1 !実際の形状を表示する")
    f.write("\n")

    f.write('finish')
    f.write("\n")

    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("dk,{},all,0".format(i + 1))
        f.write("\n")
    # 外力設定
    for i, input_vector in enumerate(input_vectors):
        f.write("fk,{},FX,{}".format(input_nodes[i] + 1, input_vector[0]))
        f.write("\n")
        f.write("fk,{},FY,{}".format(input_nodes[i] + 1, input_vector[1]))
        f.write("\n")
    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def convert_graph_info_to_mapdl_outer(nodes_pos, edges_indices, edges_thickness, input_nodes,
                                      input_vectors, frozen_nodes, b, daihenkei=True):
    path = 'haetorigusa.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,45")
    f.write("\n")
    f.write("mp,ex,1,10 ! sets modulus of mtl 1")
    f.write("\n")
    f.write("mp,prxy,1,0.3")
    f.write("\n")
    f.write("mat,1")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("k,{},{},{}".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("k,{},{},{},{}".format(i + 1, node_pos[0], node_pos[1], 0.5))
        f.write("\n")

    f.write("a,1,3,8,6")
    f.write("\n")
    f.write("a,3,4,9,8")
    f.write("\n")
    f.write("a,4,5,10,9")
    f.write("\n")
    f.write("allsel")
    f.write("\n")
    f.write("vext,all,,,0,0,5")
    f.write("\n")
    f.write("nummrg,kp,1e-6")
    f.write("\n")
    f.write("vmesh,all")
    f.write("\n")

    """
    # エッジ部分の設定
    for i, edges_indice in enumerate(edges_indices):
        h = edges_thickness[i]
        f.write("sectype,{},beam,rect,,0".format(i + 1))
        f.write("\n")
        f.write("secdata,{},{},2,2".format(h, b))  # 高さ，幅，高さに沿ったセル数，幅に沿ったセル数
        f.write("\n")

        f.write("l,{},{}".format(edges_indice[0] + 1, edges_indice[1] + 1))
        f.write("\n")
        f.write("lesize,{},,,5 !lineを10分割でmeshする".format(i + 1))
        f.write("\n")
        f.write("latt,1,1,1,,,,{} !最後の数値でsecnumを指定する".format(i + 1))
        f.write("\n")
        #f.write("r,{},{},{},{},0".format(i + 1, b * h, b * (h * h**2) / 12, h))
        # f.write("\n")
    f.write("lmesh,ALL")
    f.write("\n")
    """

    if daihenkei:
        f.write("NLGEOM,1")
        f.write("\n")
    f.write("/eshape,1 !実際の形状を表示する")
    f.write("\n")

    f.write('finish')
    f.write("\n")

    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("dk,{},all,0".format(i + 1))
        f.write("\n")
    # 外力設定
    for i, input_vector in enumerate(input_vectors):
        f.write("fk,{},FX,{}".format(input_nodes[i] + 1, input_vector[0]))
        f.write("\n")
        f.write("fk,{},FY,{}".format(input_nodes[i] + 1, input_vector[1]))
        f.write("\n")
        f.write("fk,{},FZ,{}".format(input_nodes[i] + 1, 0))
        f.write("\n")
    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def convert_graph_info_to_mapdl_code_solsh190(nodes_pos, frozen_nodes, b, pressure, daihenkei=True, substep=1e3, highest_step=1e3, min_step=1e1):
    path = 'haetorigusa_shell.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,SOLSH190 ")
    f.write("\n")

    # 材料物性1 下側の層
    f.write("MAT,1")
    f.write("\n")
    f.write("MPTEMP,1,0 ")
    f.write("\n")
    f.write("MPDATA,EX,1,,10 ")
    f.write("\n")
    f.write("MPDATA,PRXY,1,,0.3  ")
    f.write("\n")
    # 材料物性2 上側の層
    f.write("MAT,2")
    f.write("\n")
    f.write("MPTEMP,1,0 ")
    f.write("\n")
    f.write("MPDATA,EX,2,,10 ")
    f.write("\n")
    f.write("MPDATA,PRXY,2,,0.3  ")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(nodes_pos):
        f.write("n,{},{},{},0".format(i + 1, node_pos[0], node_pos[1]))
        f.write("\n")
    for i, node_pos in enumerate(nodes_pos):
        f.write("n,{},{},{},{}".format(i + 1 + nodes_pos.shape[0], node_pos[0], node_pos[1], b))
        f.write("\n")

    # シェル要素内部の構成
    f.write("sect,1,shell,,  ")
    f.write("\n")
    f.write("secdata, 0.5,1,0.0,3")
    f.write("\n")
    f.write("secdata, 0.5,2,0.0,3")
    f.write("\n")
    f.write("secoffset,MID   ")
    f.write("\n")
    f.write("seccontrol,,,, , , ,")
    f.write("\n")

    f.write("e,1,2,12,11,6,7,17,16")
    f.write("\n")
    f.write("e,2,3,13,12,7,8,18,17")
    f.write("\n")
    f.write("e,3,4,14,13,8,9,19,18")
    f.write("\n")
    f.write("e,4,5,15,14,9,10,20,19")
    f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")

    f.write('finish')
    f.write("\n")
    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for i in frozen_nodes:
        f.write("d,{},all,0".format(i + 1))
        f.write("\n")
    for i in frozen_nodes:
        f.write("d,{},all,0".format(i + 1 + nodes_pos.shape[0]))
        f.write("\n")

    # 外力設定
    f.write("ESEL,ALL")
    f.write("\n")
    f.write("SFCONTROL,0 !これをすることで，要素のもつ座標系を使用することを宣言する")
    f.write("\n")
    f.write("SFE,ALL,1,PRES,,-{} !上方向面圧".format(pressure))
    f.write("\n")
    f.write("SFE,ALL,6,PRES,,-{} !下方向面圧".format(pressure))
    f.write("\n")

    # 右方向に圧力あると下方向に移動する
    f.write("ESEL,S,ELEM,,{},{}".format(4, 4))  # 一番右端の要素の右面に圧力設定
    f.write("\n")
    f.write("SFE,ALL,3,PRES,,-{} !右方向面圧".format(pressure))
    f.write("\n")

    f.write("ESEL,ALL")
    f.write("\n")

    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def make_curvature_shape_mapdl_code_solsh190(vertical_division, horizontal_division, pressure, vertical_length=6.5, horizontal_length=15,
                                             curvature=0.04, left_thick=1, right_thick=0.5, daihenkei=True, substep=1e3, highest_step=1e3, min_step=1e1):
    lower_nodes_pos, higher_nodes_pos = make_node(vertical_division, horizontal_division, vertical_length, horizontal_length,
                                                  curvature, left_thick, right_thick)

    path = 'haetorigusa_shell.apdl'
    f = open(path, 'w')
    f.write('finish')
    f.write("\n")
    f.write('/clear')
    f.write("\n")
    f.write('/prep7')
    f.write("\n")

    f.write("et,1,SOLSH190 ")
    f.write("\n")

    # 材料物性1 下側の層
    f.write("MAT,1")
    f.write("\n")
    f.write("MPTEMP,1,0 ")
    f.write("\n")
    f.write("MPDATA,EX,1,,10 ")
    f.write("\n")
    f.write("MPDATA,PRXY,1,,0.3  ")
    f.write("\n")
    # 材料物性2 上側の層
    f.write("MAT,2")
    f.write("\n")
    f.write("MPTEMP,1,0 ")
    f.write("\n")
    f.write("MPDATA,EX,2,,10 ")
    f.write("\n")
    f.write("MPDATA,PRXY,2,,0.3  ")
    f.write("\n")

    # 節点部分の設定
    for i, node_pos in enumerate(lower_nodes_pos):
        f.write("n,{},{},{},{}".format(i + 1, node_pos[0], node_pos[1], node_pos[2]))
        f.write("\n")
    for i, node_pos in enumerate(higher_nodes_pos):
        f.write("n,{},{},{},{}".format(i + 1 + lower_nodes_pos.shape[0], node_pos[0], node_pos[1], node_pos[2]))
        f.write("\n")

    # シェル要素内部の構成
    f.write("sect,1,shell,,  ")
    f.write("\n")
    f.write("secdata, 0.5,1,0.0,3")
    f.write("\n")
    f.write("secdata, 0.5,2,0.0,3")
    f.write("\n")
    f.write("secoffset,MID   ")
    f.write("\n")
    f.write("seccontrol,,,, , , ,")
    f.write("\n")

    lower_point_num = lower_nodes_pos.shape[0]
    for vt in range(vertical_division - 1):
        for ht in range(horizontal_division - 1):
            [e1, e2, e3, e4] = [vt * horizontal_division + ht + 1, vt * horizontal_division + ht + 2,
                                (vt + 1) * horizontal_division + ht + 2, (vt + 1) * horizontal_division + ht + 1]
            f.write("e,{},{},{},{},{},{},{},{}".format(e1, e2, e3, e4,
                                                       e1 + lower_point_num, e2 + lower_point_num, e3 + lower_point_num, e4 + lower_point_num))
            f.write("\n")

    if daihenkei:
        f.write("AUTOTS,on")
        f.write("\n")
        f.write("NLGEOM,1")
        f.write("\n")
        f.write("NSUBST,{},{},{}".format(substep, highest_step, min_step))
        f.write("\n")

    f.write('finish')
    f.write("\n")
    f.write('/solu')
    f.write("\n")
    f.write('static')
    f.write("\n")

    # 解析条件設定
    # 固定ノード設定
    for ht in range(horizontal_division):
        f.write("d,{},all,0".format(ht + 1))
        f.write("\n")
        f.write("d,{},all,0".format(ht + 1 + lower_point_num))
        f.write("\n")

    # 外力設定
    f.write("ESEL,ALL")
    f.write("\n")
    f.write("SFCONTROL,0 !これをすることで，要素のもつ座標系を使用することを宣言する")
    f.write("\n")
    f.write("SFE,ALL,1,PRES,,-{} !上方向面圧".format(pressure))
    f.write("\n")
    f.write("SFE,ALL,6,PRES,,-{} !下方向面圧".format(pressure))
    f.write("\n")

    # 右方向に圧力あると下方向に移動する
    for he in range(horizontal_division - 1):
        elem_num = (he + 1) + (horizontal_division - 1) * (vertical_division - 2)
        f.write("ESEL,S,ELEM,,{},{}".format(elem_num, elem_num))  # 一番右端の要素の右面に圧力設定
        f.write("\n")
        f.write("SFE,ALL,4,PRES,,-{} !右方向面圧".format(pressure))
        f.write("\n")

    f.write("ESEL,ALL")
    f.write("\n")

    # 解析開始
    f.write('solve')
    f.write("\n")
    f.write('finish')
    f.write("\n")

    f.close()


def make_calc_efficiency_with_ansys_python_code(path):
    # ANSYSのηを計算する為のpythonコード作成
    f = open(path, 'r')
    datalist = f.readlines()
    f.close()

    path = 'mapdl.py'
    f = open(path, 'w')
    f.write("from ansys.mapdl.core import launch_mapdl")
    f.write("\n")
    f.write("import numpy as np")
    f.write("\n")
    f.write("from GA.condition import venus_trap_condition")
    f.write("\n")
    f.write("from ansys.mapdl.core import launch_mapdl")
    f.write("\n")
    f.write("from tools.graph import *")
    f.write("\n")
    f.write("mapdl = launch_mapdl()")
    f.write("\n")
    for data in datalist:
        f.write("mapdl.run(\"" + data.rstrip('\n') + "\")")
        f.write("\n")
    f.write("x_disp = mapdl.post_processing.nodal_displacement('X')\n")
    f.write("y_disp = mapdl.post_processing.nodal_displacement('Y')\n")
    f.write("z_rot = mapdl.post_processing.nodal_rotation('Z')\n")
    f.write("displacement = np.stack([x_disp, y_disp, z_rot]).T.flatten()\n")
    f.write("_, input_nodes, input_vectors, output_nodes, output_vectors,_, _, _, L, A = venus_trap_condition(b={}, edge_width=0.025 * {})\n".format(b, edge_thick_big_size))
    f.write("efficiency = calc_output_efficiency(input_nodes, input_vectors * {}, output_nodes, output_vectors, displacement, E={}, A=A, L=L)\n".format(pressure, E))
    f.write("print(efficiency)\n")
    f.close()
