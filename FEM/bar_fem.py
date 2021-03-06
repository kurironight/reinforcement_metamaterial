import ctypes as ct
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy as np

libc = ct.cdll.LoadLibrary("FEM/barfem.so")


def barfem(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes, mode="displacement"):
    """バーFEMを行う

    Args:
        nodes_pos (np.array): node_num*2.ノードの位置座標を収納
        edges_indices (np.array): エッジの繋がりを示している．なお，ノード数が5この時，[[0,1],[2,3]]と4が含まれないなどのようなことはないものとする．
        edges_thickness (np.array): 各エッジの太さを示している．
        input_nodes (list): 変位を入力するノードを指定している
        input_vectors (np.array): 入力する変位(or 力)を指定している
        frozen_nodes (list)): 固定しているノードを指定している
        mode(str): 強制変位か外力の条件の時の有限要素法を扱う．強制変位の時は"displacement",外力の時は"force"とする．

    Returns:
        [np.array]: 各要素の変位を示している．
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
    displacement = np.ones((node_num * 3,))  # 各節点要素の変位を持つ変数

    assert input_vectors.shape[1] == 2, '求められている強制変位もしくは外力はx,y方向のみである'

    assert np.all(np.isin(np.arange(node_num), edges_indices)
                  ), 'edge_indicesでは触れられていないノードが存在する'
    assert mode == 'displacement' or mode == 'force', 'modeは"displacement"か"force"'

    # doubleのポインタのポインタ型を用意
    _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # 関数の引数の型を指定(ctypes)　
    libc.bar_fem.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                             ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP]
    libc.bar_fem_force.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32, ct.c_int32,
                                   ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ct.c_int32, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP]
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

    inp = (ctypes.c_int * len(input_nodes))(*input_nodes)
    frz = (ctypes.c_int * len(frozen_nodes))(*frozen_nodes)
    if mode == 'displacement':
        libc.bar_fem(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                     frz, dspp)
    elif mode == 'force':
        libc.bar_fem_force(mpp, eipp, etpp, cnode_num, cedge_num, cinput_node_num, inp, ivpp, cfrozen_node_num,
                           frz, dspp)

    return displacement
