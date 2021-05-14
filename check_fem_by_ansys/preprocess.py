import numpy as np

max_node_num = 4
max_edge_num = 3

assert max_edge_num > 2, "max_edge_numは3以上である必要あり"

flag = 0

while flag == 0:
    node_num = np.random.randint(2, max_node_num)
    edge_num = np.random.randint(1, max_edge_num)

    edges_indices = np.random.randint(0, node_num, (edge_num, 2))
    edges_indices = edges_indices[edges_indices[:, 0] != edges_indices[:, 1]]  # [1,1]などの組み合わせを削除
    if edges_indices.shape[0] == 0:
        continue
    edges_indices = np.sort(edges_indices, axis=1)
    print(edges_indices)
    edges_indices = np.unique(edges_indices, axis=0)  # 重複を削除

    while edges_indices.shape[0] == 0:
        edges_indices = np.random.randint(0, node_num, (edge_num, 2))
        edges_indices = edges_indices[edges_indices[:, 0] != edges_indices[:, 1]]
        edges_indices = np.sort(edges_indices, axis=1)
        edges_indices = np.unique(edges_indices, axis=0)
    ref_edges_indices = edges_indices.copy()

    exist_nodes = np.unique(edges_indices)
    node_num = exist_nodes.shape[0]
    if node_num >= 2:
        flag = 1
    else:
        continue
    frozen_num = np.random.randint(1, node_num)
    edge_num = edges_indices.shape[0]
    ref_nodes = np.arange(node_num)

    for exist_node, ref_node in zip(exist_nodes, ref_nodes):
        edges_indices[ref_edges_indices == exist_node] = ref_node

    # ノードの位置およびエッジの太さをランダム設定
    nodes_pos = np.random.uniform(0, 1, (node_num, 2))
    edges_thickness = np.random.uniform(0, 1, edge_num)

    # 固定ノードを選択
    frozen_nodes = np.unique(np.random.randint(0, node_num, frozen_num)).tolist()
    input_nodes = np.random.randint(0, node_num, 1).tolist()
    while input_nodes[0] in frozen_nodes:
        input_nodes = np.random.randint(0, node_num, 1).tolist()
print(edges_indices)
print(frozen_nodes)
print(input_nodes)
