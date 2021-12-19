# 段階的GAにおいて，各ノード数においてのノードの構造の変化を確認する．

import numpy as np
import os
import networkx as nx
from tqdm import tqdm

data_num = 5
interval = 5
first_data = 5  # 各段階的GAで保存されている最初のepoch
last_data = 115  # 各段階的GAで保存されている最後のepoch
save_epochs = np.arange(first_data, last_data + 1, interval)
max_node_num = 6
load_dir = "//ZUIHO/share/user/knakamur/Metamaterial/seminar_data/11_11data/中間エッジ追加型GA_n6_固定有"

Jaccard_list = []


def jaccard_similarity(G1_edge, G2_edge):
    over_lap = set(G1_edge).intersection(G2_edge)
    return len(over_lap) / (len(G1_edge) + len(G2_edge) - len(over_lap))


def make_G(dir):
    nodes_pos = np.load(os.path.join(dir, 'nodes_pos.npy'))
    edges_indices = np.load(os.path.join(dir, 'edges_indices.npy'))

    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes_pos)))
    G.add_edges_from(edges_indices)
    return G


for i in tqdm(range(data_num)):
    data_Jaccard_list = []
    for node in range(max_node_num):
        node = node + 1
        flag = True
        for j in range(len(save_epochs) - 1):
            dir = os.path.join(load_dir, "{}/free_{}_fix_8/{}".format(i, node, save_epochs[j]))
            if not os.path.exists(os.path.join(dir, 'nodes_pos.npy')):
                continue

            if flag:
                G1 = make_G(dir)
                dir = os.path.join(load_dir, "{}/free_{}_fix_8/{}".format(i, node, save_epochs[j + 1]))
                G2 = make_G(dir)
                flag = False
            else:
                G1 = G2
                dir = os.path.join(load_dir, "{}/free_{}_fix_8/{}".format(i, node, save_epochs[j + 1]))
                G2 = make_G(dir)

            minv = jaccard_similarity(G1.edges(), G2.edges())
            data_Jaccard_list.append(minv)
    Jaccard_list.append(data_Jaccard_list)

np.save(os.path.join(load_dir, "Jaccard.npy"), Jaccard_list)

Jaccard_list = np.load(os.path.join(load_dir, "Jaccard.npy"), allow_pickle=True)
for i in Jaccard_list:
    print(np.mean(i))
