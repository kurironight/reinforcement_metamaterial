import ctypes as ct
from ctypes import *
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy as np
import time

libc = ct.cdll.LoadLibrary("./libmyadd.so")

# 適当につくります
node_num = 5
edge_num = 4

nodes_pos = np.random.rand(node_num, 2)
# edges_indices = np.random.randint(0, node_num, (edge_num, 2), dtype=np.int32)
edges_indices = np.array([
    [0, 1],
    [1, 2],
    [1, 3],
    [1, 4],
], dtype=np.int32)
edges_thickness = np.random.rand(node_num, )
input_nodes = [0, 1]
input_vectors = np.array([
    [0., -0.2],
    [0., -0.2],
])
frozen_nodes = [3, 4]

# doubleのポインタのポインタ型を用意
_DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
# add_matrix()関数の引数の型を指定(ctypes)　
libc.kata.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, c_int32, c_int32,
                      ctypes.POINTER(ctypes.c_int), _DOUBLE_PP, ctypes.POINTER(ctypes.c_int)]
# add_matrix()関数が返す値の型を指定(今回は返り値なし)
libc.kata.restype = None

# 各配列のアドレスを求めている
tp = np.uintp

mpp = (nodes_pos.__array_interface__[
       'data'][0] + np.arange(nodes_pos.shape[0])*nodes_pos.strides[0]).astype(tp)
eipp = (edges_indices .__array_interface__[
    'data'][0] + np.arange(edges_indices .shape[0])*edges_indices.strides[0]).astype(tp)
etpp = (edges_thickness.__array_interface__[
    'data'][0] + np.arange(edges_thickness.shape[0])*edges_thickness.strides[0]).astype(tp)
ivpp = (input_vectors.__array_interface__[
    'data'][0] + np.arange(input_vectors.shape[0])*input_vectors.strides[0]).astype(tp)

# int型もctypeのc_int型へ変換して渡す
cnode_num = ctypes.c_int(node_num)
cedge_num = ctypes.c_int(edge_num)

inp = (ctypes.c_int*len(input_nodes))(*input_nodes)
frz = (ctypes.c_int*len(frozen_nodes))(*frozen_nodes)


start = time.time()

#print("before:", edges_indices)

libc.kata(mpp, eipp, etpp, cnode_num, cedge_num, inp, ivpp, frz)

#print("after:", edges_indices)

t = time.time() - start

print("時間; ", t)
