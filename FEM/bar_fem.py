import ctypes as ct
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy as np
from tools.graph import calc_length

libc = ct.cdll.LoadLibrary("FEM/barfem.so")
libd = ct.cdll.LoadLibrary("FEM/barfem_anti.so")


def barfem(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, mode="displacement", slender=False, tmax=100000, eps=1.0e-11):
    """バーFEMを行う

    Args:
        nodes_pos (np.array): node_num*2.ノードの位置座標を収納
        edges_indices (np.array): エッジの繋がりを示している．なお，ノード数が5この時，[[0,1],[2,3]]と4が含まれないなどのようなことはないものとする．
        edges_thickness (np.array): 各エッジの太さを示している．
        input_nodes (list): 変位を入力するノードを指定している
        input_vectors (np.array): 入力する変位(or 力)を指定している
        frozen_nodes (list)): 固定しているノードを指定している
        tmax (int, optional): 共益勾配法のステップ数. Defaults to 100000.
        eps (double, optional): 共益勾配法の収束条件. Defaults to 1.0e-11.
        mode(str): 強制変位か外力の条件の時の有限要素法を扱う．強制変位の時は"displacement",外力の時は"force"とする．

    Returns:
        [np.array]: 各要素の変位を示している．
    """

    if slender:  # スレンダー比を考慮し，edge_thicknessが各エッジの太さに対して1/10を守るように設定する
        edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
        lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])
        edges_thickness = edges_thickness * lengths / 10

    node_num = nodes_pos.shape[0]
    edge_num = edges_indices.shape[0]
    input_node_num = len(input_nodes)
    frozen_node_num = len(frozen_nodes)
    nodes_pos = nodes_pos.astype(
        np.float64)
    input_vectors = input_vectors.astype(
        np.float64)
    edges_indices = edges_indices.astype(
        np.int32)  # ここをint32型にしないとコードが正しく作動しない
    edges_thickness = edges_thickness.astype(
        np.float64)  # ここをfloat64型にしないとコードが正しく作動しない
    displacement = np.ones((node_num * 3,))  # 各節点要素の変位を持つ変数
    tmax = int(tmax)
    eps = float(eps)

    assert edges_indices.shape[0] == edges_thickness.shape[0], 'edges_thicknessとedges_indicesは同じ数あるべきである'
    assert input_vectors.shape[1] == 2, 'input_vectorsの形は[[[0, -1]],[0,3]]のような形である'
    assert edges_indices.shape[1] == 2, 'edges_indicesの形は[[0,1],[2,3]]のような形である'

    assert np.all(np.isin(np.arange(node_num), edges_indices)
                  ), 'edge_indicesでは触れられていないノードが存在する'
    assert mode == 'displacement' or mode == 'force', 'modeは"displacement"か"force"'
    assert np.unique(nodes_pos, axis=0).shape[0] == nodes_pos.shape[0], "同じ座標を示すノードがnode_posに含まれている"
    assert np.unique(edges_indices, axis=0).shape[0] == edges_indices.shape[0], "同じedge_indiceがedge_indicesに含まれている"
    assert np.max(input_nodes) < node_num, "input_nodesに含まれる指定ノードがノード数を超えている"
    assert np.max(frozen_nodes) < node_num, "frozen_nodesに含まれる指定ノードがノード数を超えている"

    # doubleのポインタのポインタ型を用意
    _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # 関数の引数の型を指定(ctypes)　
    libc.bar_fem.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                             ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ct.c_double]
    libc.bar_fem_force.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                                   ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ct.c_double]
    # 関数が返す値の型を指定(今回は返り値なし)
    libc.bar_fem.restype = None
    libc.bar_fem_force.restype = None

    # 各配列のアドレスを求めている
    tp = np.uintp

    mpp = (nodes_pos.__array_interface__[
           'data'][0] + np.arange(nodes_pos.shape[0]) * nodes_pos.strides[0]).astype(tp)
    eipp = (edges_indices .__array_interface__[
        'data'][0] + np.arange(edges_indices .shape[0]) * edges_indices.strides[0]).astype(tp)
    etpp = (edges_thickness.__array_interface__[
        'data'][0] + np.arange(edges_thickness.shape[0]) * edges_thickness.strides[0]).astype(tp)
    dspp = (displacement.__array_interface__[
        'data'][0] + np.arange(displacement.shape[0]) * displacement.strides[0]).astype(tp)
    ivpp = (input_vectors.__array_interface__[
        'data'][0] + np.arange(input_vectors.shape[0]) * input_vectors.strides[0]).astype(tp)

    # int型もctypeのc_int型へ変換して渡す
    cnode_num = ctypes.c_int(node_num)
    cedge_num = ctypes.c_int(edge_num)
    cinput_node_num = ctypes.c_int(input_node_num)
    cfrozen_node_num = ctypes.c_int(frozen_node_num)
    ctmax = ctypes.c_int(tmax)
    ceps = ctypes.c_double(eps)

    inp = (ctypes.c_int * len(input_nodes))(*input_nodes)
    frz = (ctypes.c_int * len(frozen_nodes))(*frozen_nodes)
    if mode == 'displacement':
        libc.bar_fem(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                     frz, dspp, ctmax, ceps)
    elif mode == 'force':
        libc.bar_fem_force(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                           frz, dspp, ctmax, ceps)

    return displacement


def confirm_apdl_accuracy(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, displacement):
    """APDLのソルバーの精度を測る.
    displacementには，apdlで出力された変位を利用する．
    """
    node_num = nodes_pos.shape[0]
    edge_num = edges_indices.shape[0]
    input_node_num = len(input_nodes)
    frozen_node_num = len(frozen_nodes)
    nodes_pos = nodes_pos.astype(
        np.float64)
    input_vectors = input_vectors.astype(
        np.float64)
    edges_indices = edges_indices.astype(
        np.int32)  # ここをint32型にしないとコードが正しく作動しない
    edges_thickness = edges_thickness.astype(
        np.float64)  # ここをfloat64型にしないとコードが正しく作動しない

    assert input_vectors.shape[1] == 2, 'input_vectorsの形は[[[0, -1]],[0,3]]のような形である'
    assert edges_indices.shape[1] == 2, 'edges_indicesの形は[[0,1],[2,3]]のような形である'

    assert np.all(np.isin(np.arange(node_num), edges_indices)
                  ), 'edge_indicesでは触れられていないノードが存在する'

    # doubleのポインタのポインタ型を用意
    _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # 関数の引数の型を指定(ctypes)　
    libc.confirm_apdl_accuracy.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                                           ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP]
    # 関数が返す値の型を指定(今回は返り値なし)
    libc.confirm_apdl_accuracy.restype = None

    # 各配列のアドレスを求めている
    tp = np.uintp

    mpp = (nodes_pos.__array_interface__[
           'data'][0] + np.arange(nodes_pos.shape[0]) * nodes_pos.strides[0]).astype(tp)
    eipp = (edges_indices .__array_interface__[
        'data'][0] + np.arange(edges_indices .shape[0]) * edges_indices.strides[0]).astype(tp)
    etpp = (edges_thickness.__array_interface__[
        'data'][0] + np.arange(edges_thickness.shape[0]) * edges_thickness.strides[0]).astype(tp)
    dspp = (displacement.__array_interface__[
        'data'][0] + np.arange(displacement.shape[0]) * displacement.strides[0]).astype(tp)
    ivpp = (input_vectors.__array_interface__[
        'data'][0] + np.arange(input_vectors.shape[0]) * input_vectors.strides[0]).astype(tp)

    # int型もctypeのc_int型へ変換して渡す
    cnode_num = ctypes.c_int(node_num)
    cedge_num = ctypes.c_int(edge_num)
    cinput_node_num = ctypes.c_int(input_node_num)
    cfrozen_node_num = ctypes.c_int(frozen_node_num)

    inp = (ctypes.c_int * len(input_nodes))(*input_nodes)
    frz = (ctypes.c_int * len(frozen_nodes))(*frozen_nodes)
    libc.confirm_apdl_accuracy(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                               frz, dspp)


def barfem_mapdl(mapdl, nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, mode="displacement", slender=False):
    """バーFEMを行う

    Args:
        nodes_pos (np.array): node_num*2.ノードの位置座標を収納
        edges_indices (np.array): エッジの繋がりを示している．なお，ノード数が5この時，[[0,1],[2,3]]と4が含まれないなどのようなことはないものとする．
        edges_thickness (np.array): 各エッジの太さを示している．
        input_nodes (list): 変位を入力するノードを指定している
        input_vectors (np.array): 入力する変位(or 力)を指定している
        frozen_nodes (list)): 固定しているノードを指定している
        tmax (int, optional): 共益勾配法のステップ数. Defaults to 100000.
        eps (double, optional): 共益勾配法の収束条件. Defaults to 1.0e-11.

    Returns:
        [np.array]: 各要素の変位を示している．
    """

    if slender:  # スレンダー比を考慮し，edge_thicknessが各エッジの太さに対して1/10を守るように設定する
        edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
        lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])
        edges_thickness = edges_thickness * lengths / 10

    node_num = nodes_pos.shape[0]
    nodes_pos = nodes_pos.astype(
        np.float64)
    input_vectors = input_vectors.astype(
        np.float64)
    edges_indices = edges_indices.astype(
        np.int32)  # ここをint32型にしないとコードが正しく作動しない
    edges_thickness = edges_thickness.astype(
        np.float64)  # ここをfloat64型にしないとコードが正しく作動しない
    displacement = np.ones((node_num * 3,))  # 各節点要素の変位を持つ変数

    assert edges_indices.shape[0] == edges_thickness.shape[0], 'edges_thicknessとedges_indicesは同じ数あるべきである'
    assert input_vectors.shape[1] == 2, 'input_vectorsの形は[[[0, -1]],[0,3]]のような形である'
    assert edges_indices.shape[1] == 2, 'edges_indicesの形は[[0,1],[2,3]]のような形である'

    assert np.all(np.isin(np.arange(node_num), edges_indices)
                  ), 'edge_indicesでは触れられていないノードが存在する'
    assert mode == 'displacement' or mode == 'force', 'modeは"displacement"か"force"'
    assert np.unique(nodes_pos, axis=0).shape[0] == nodes_pos.shape[0], "同じ座標を示すノードがnode_posに含まれている"
    assert np.unique(edges_indices, axis=0).shape[0] == edges_indices.shape[0], "同じedge_indiceがedge_indicesに含まれている"
    assert np.max(input_nodes) < node_num, "input_nodesに含まれる指定ノードがノード数を超えている"
    assert np.max(frozen_nodes) < node_num, "frozen_nodesに含まれる指定ノードがノード数を超えている"

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
        mapdl.d(input_nodes[i] + 1, "UX", input_vector[0])
        mapdl.d(input_nodes[i] + 1, "UY", input_vector[1])
    # 解析開始
    mapdl.solve()
    mapdl.finish()

    # 結果出力
    x_disp = mapdl.post_processing.nodal_displacement('X')
    y_disp = mapdl.post_processing.nodal_displacement('Y')
    z_rot = mapdl.post_processing.nodal_rotation('Z')

    displacement = np.stack([x_disp, y_disp, z_rot]).T.flatten()

    return displacement


def barfem_anti(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, mode="displacement", slender=False, tmax=100000, eps=1.0e-11):
    """バーFEMを行う

    Args:
        nodes_pos (np.array): node_num*2.ノードの位置座標を収納
        edges_indices (np.array): エッジの繋がりを示している．なお，ノード数が5この時，[[0,1],[2,3]]と4が含まれないなどのようなことはないものとする．
        edges_thickness (np.array): 各エッジの太さを示している．
        input_nodes (list): 変位を入力するノードを指定している
        input_vectors (np.array): 入力する変位(or 力)を指定している
        frozen_nodes (list)): 固定しているノードを指定している
        tmax (int, optional): 共益勾配法のステップ数. Defaults to 100000.
        eps (double, optional): 共益勾配法の収束条件. Defaults to 1.0e-11.
        mode(str): 強制変位か外力の条件の時の有限要素法を扱う．強制変位の時は"displacement",外力の時は"force"とする．

    Returns:
        [np.array]: 各要素の変位を示している．
    """

    if slender:  # スレンダー比を考慮し，edge_thicknessが各エッジの太さに対して1/10を守るように設定する
        edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
        lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])
        edges_thickness = edges_thickness * lengths / 10

    node_num = nodes_pos.shape[0]
    edge_num = edges_indices.shape[0]
    input_node_num = len(input_nodes)
    frozen_node_num = len(frozen_nodes)
    nodes_pos = nodes_pos.astype(
        np.float64)
    input_vectors = input_vectors.astype(
        np.float64)
    edges_indices = edges_indices.astype(
        np.int32)  # ここをint32型にしないとコードが正しく作動しない
    edges_thickness = edges_thickness.astype(
        np.float64)  # ここをfloat64型にしないとコードが正しく作動しない
    displacement = np.ones((node_num * 3,))  # 各節点要素の変位を持つ変数
    stresses = np.ones((edge_num * 6,))  # 各節点要素の応力を持つ変数
    tmax = int(tmax)
    eps = float(eps)

    assert edges_indices.shape[0] == edges_thickness.shape[0], 'edges_thicknessとedges_indicesは同じ数あるべきである'
    assert input_vectors.shape[1] == 2, 'input_vectorsの形は[[[0, -1]],[0,3]]のような形である'
    assert edges_indices.shape[1] == 2, 'edges_indicesの形は[[0,1],[2,3]]のような形である'

    assert np.all(np.isin(np.arange(node_num), edges_indices)
                  ), 'edge_indicesでは触れられていないノードが存在する'
    assert mode == 'displacement' or mode == 'force', 'modeは"displacement"か"force"'
    assert np.unique(nodes_pos, axis=0).shape[0] == nodes_pos.shape[0], "同じ座標を示すノードがnode_posに含まれている"
    assert np.unique(edges_indices, axis=0).shape[0] == edges_indices.shape[0], "同じedge_indiceがedge_indicesに含まれている"
    assert np.max(input_nodes) < node_num, "input_nodesに含まれる指定ノードがノード数を超えている"
    assert np.max(frozen_nodes) < node_num, "frozen_nodesに含まれる指定ノードがノード数を超えている"

    # doubleのポインタのポインタ型を用意
    _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # 関数の引数の型を指定(ctypes)　
    libd.bar_fem.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                             ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_double]
    libd.bar_fem_force.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                                   ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_double]
    # 関数が返す値の型を指定(今回は返り値なし)
    libd.bar_fem.restype = None
    libd.bar_fem_force.restype = None

    # 各配列のアドレスを求めている
    tp = np.uintp

    mpp = (nodes_pos.__array_interface__[
           'data'][0] + np.arange(nodes_pos.shape[0]) * nodes_pos.strides[0]).astype(tp)
    eipp = (edges_indices .__array_interface__[
        'data'][0] + np.arange(edges_indices .shape[0]) * edges_indices.strides[0]).astype(tp)
    etpp = (edges_thickness.__array_interface__[
        'data'][0] + np.arange(edges_thickness.shape[0]) * edges_thickness.strides[0]).astype(tp)
    dspp = (displacement.__array_interface__[
        'data'][0] + np.arange(displacement.shape[0]) * displacement.strides[0]).astype(tp)
    stpp = (stresses.__array_interface__[
        'data'][0] + np.arange(stresses.shape[0]) * stresses.strides[0]).astype(tp)
    ivpp = (input_vectors.__array_interface__[
        'data'][0] + np.arange(input_vectors.shape[0]) * input_vectors.strides[0]).astype(tp)

    # int型もctypeのc_int型へ変換して渡す
    cnode_num = ctypes.c_int(node_num)
    cedge_num = ctypes.c_int(edge_num)
    cinput_node_num = ctypes.c_int(input_node_num)
    cfrozen_node_num = ctypes.c_int(frozen_node_num)
    ctmax = ctypes.c_int(tmax)
    ceps = ctypes.c_double(eps)

    inp = (ctypes.c_int * len(input_nodes))(*input_nodes)
    frz = (ctypes.c_int * len(frozen_nodes))(*frozen_nodes)
    if mode == 'displacement':
        libd.bar_fem(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                     frz, dspp, stpp, ctmax, ceps)
    elif mode == 'force':
        libd.bar_fem_force(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                           frz, dspp, stpp, ctmax, ceps)
    return displacement, stresses


"""
使用例

from FEM.bar_fem import barfem
import numpy as np

nodes_pos = np.array(
    [[0, 0], [5, 0]]
)
edges_indices = np.array([[0, 1]])
edges_thickness = np.array([2])
input_nodes = [1]
input_vectors = np.array([[0, -1000]])
frozen_nodes = [0]

displacement = barfem(nodes_pos, edges_indices, edges_thickness, input_nodes,
                      input_vectors, frozen_nodes, mode="force")

print(displacement)
"""
