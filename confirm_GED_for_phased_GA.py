# 段階的GAにおいて，各ノード数においてのノードの構造の変化を確認する．

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tools.graph import *
from FEM.bar_fem import barfem_anti
import networkx as nx

data_num = 5
first_data = 5  # 各段階的GAで保存されている最初のepoch
last_data = 115  # 各段階的GAで保存されている最後のepoch
max_node_num = 6
load_dir = "//ZUIHO/share/user/knakamur/Metamaterial/seminar_data/11_11data/新規エッジ追加型GA_n6_固定有"

GED_list = []
for i in range(data_num):
    data_GED_list = []
    for node in range(max_node_num):
        node = node + 1
        dir = os.path.join(load_dir, "{}/free_{}_fix_8/{}".format(i, node, first_data))

        nodes_pos = np.load(os.path.join(dir, 'nodes_pos.npy'))
        edges_indices = np.load(os.path.join(dir, 'edges_indices.npy'))
        edges_thickness = np.load(os.path.join(dir, 'edges_thickness.npy'))
        input_nodes = np.load(os.path.join(dir, 'input_nodes.npy')).tolist()
        input_vectors = np.load(os.path.join(dir, 'input_vectors.npy'))
        frozen_nodes = np.load(os.path.join(dir, 'frozen_nodes.npy')).tolist()
        output_nodes = np.load(os.path.join(dir, 'output_nodes.npy')).tolist()
        output_vectors = np.load(os.path.join(dir, 'output_vectors.npy'))

        G1 = nx.Graph()
        G1.add_nodes_from(np.arange(len(nodes_pos)))
        edge_info = np.concatenate([edges_indices, edges_thickness.reshape((-1, 1))], axis=1)
        G1.add_edges_from(edges_indices)

        dir = os.path.join(load_dir, "{}/free_{}_fix_8/{}".format(i, node, last_data))

        nodes_pos = np.load(os.path.join(dir, 'nodes_pos.npy'))
        edges_indices = np.load(os.path.join(dir, 'edges_indices.npy'))
        edges_thickness = np.load(os.path.join(dir, 'edges_thickness.npy'))
        input_nodes = np.load(os.path.join(dir, 'input_nodes.npy')).tolist()
        input_vectors = np.load(os.path.join(dir, 'input_vectors.npy'))
        frozen_nodes = np.load(os.path.join(dir, 'frozen_nodes.npy')).tolist()
        output_nodes = np.load(os.path.join(dir, 'output_nodes.npy')).tolist()
        output_vectors = np.load(os.path.join(dir, 'output_vectors.npy'))

        G2 = nx.Graph()
        G2.add_nodes_from(np.arange(len(nodes_pos)))
        edge_info = np.concatenate([edges_indices, edges_thickness.reshape((-1, 1))], axis=1)
        G2.add_edges_from(edges_indices)
        print("計算中")
        for v in nx.optimize_graph_edit_distance(G1, G2):
            minv = v
            print(minv)
        data_GED_list.append(minv)
        np.save(os.path.join(load_dir, "GED_{}_{}.npy".format(i, node)), minv)
    GED_list.append(data_GED_list)

print(GED_list)
np.save(os.path.join(load_dir, "GED.npy"), GED_list)
