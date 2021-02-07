import ctypes as ct
from ctypes import *
import ctypes.util
from numpy.ctypeslib import ndpointer
import numpy as np

libc = ct.cdll.LoadLibrary("./libmyadd.so")

# 適当につくります
node_num = 10
edge_num = 5

nodes_pos = np.random.rand(node_num, 2)
edges_indices1 = np.random.randint(0, node_num, (edge_num, 1), dtype=np.int32)
edges_thickness = np.random.rand(node_num, )

# doubleのポインタのポインタ型を用意
_DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
_INT_PP = ctypes.POINTER(ctypes.c_uint32)
# add_matrix()関数の引数の型を指定(ctypes)　
libc.kata.argtypes = [_DOUBLE_PP, _INT_PP, _DOUBLE_PP, c_int32, c_int32]
# add_matrix()関数が返す値の型を指定(今回は返り値なし)
libc.kata.restype = None

# 各配列のアドレスを求めている
tp = np.uintp

mpp = (nodes_pos.__array_interface__[
       'data'][0] + np.arange(nodes_pos.shape[0])*nodes_pos.strides[0]).astype(tp)
eipp = edges_indices1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
print(eipp)
etpp = (edges_thickness.__array_interface__[
    'data'][0] + np.arange(edges_thickness.shape[0])*edges_thickness.strides[0]).astype(tp)
# int型もctypeのc_int型へ変換して渡す
cnode_num = ctypes.c_int(node_num)
cedge_num = ctypes.c_int(edge_num)


print("before:", edges_indices1)

libc.kata(mpp, eipp, etpp, cnode_num, cedge_num)

print("after:", edges_indices1)
