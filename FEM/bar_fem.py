import ctypes as ct
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy as np

libc = ct.cdll.LoadLibrary("FEM/barfem.so")


def barfem(nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes):
    node_num = nodes_pos.shape[0]
    edge_num = edges_indices.shape[0]
    edges_indices = edges_indices.astype(
        np.int32)  # ここをint32型にしないとコードが正しく作動しない
    displacement = np.ones((node_num*3,))  # 各節点要素の変位を持つ変数

    # doubleのポインタのポインタ型を用意
    _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # 関数の引数の型を指定(ctypes)　
    libc.bar_fem.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, ct.c_int32, ct.c_int32,
                             ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ctypes.POINTER(ctypes.c_int), _DOUBLE_PP]
    # 関数が返す値の型を指定(今回は返り値なし)
    libc.bar_fem.restype = None

    # 各配列のアドレスを求めている
    tp = np.uintp

    mpp = (nodes_pos.__array_interface__[
           'data'][0] + np.arange(nodes_pos.shape[0])*nodes_pos.strides[0]).astype(tp)
    eipp = (edges_indices .__array_interface__[
        'data'][0] + np.arange(edges_indices .shape[0])*edges_indices.strides[0]).astype(tp)
    etpp = (edges_thickness.__array_interface__[
        'data'][0] + np.arange(edges_thickness.shape[0])*edges_thickness.strides[0]).astype(tp)
    dspp = (displacement.__array_interface__[
        'data'][0] + np.arange(displacement.shape[0])*displacement.strides[0]).astype(tp)
    ivpp = (input_vectors.__array_interface__[
        'data'][0] + np.arange(input_vectors.shape[0])*input_vectors.strides[0]).astype(tp)

    # int型もctypeのc_int型へ変換して渡す
    cnode_num = ctypes.c_int(node_num)
    cedge_num = ctypes.c_int(edge_num)

    inp = (ctypes.c_int*len(input_nodes))(*input_nodes)
    frz = (ctypes.c_int*len(frozen_nodes))(*frozen_nodes)

    libc.bar_fem(mpp, eipp, etpp, cnode_num, cedge_num, inp, ivpp, frz, dspp)

    return displacement
