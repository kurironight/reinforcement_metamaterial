import numpy as np
import networkx as nx


def make_random_fem_condition(max_node_num, max_edge_num):
    """ランダムにFEMの条件を生成する

    Args:
        max_node_num (int): 最大のノードの数
        max_edge_num (int): 最大のエッジの数
    """
    assert max_edge_num > 2, "max_edge_numは3以上である必要あり"

    flag = 0

    while flag == 0:
        node_num = np.random.randint(2, max_node_num)
        edge_num = np.random.randint(1, max_edge_num)

        # エッジを選定する
        edges_indices = np.random.randint(0, node_num, (edge_num, 2))
        edges_indices = edges_indices[edges_indices[:, 0] != edges_indices[:, 1]]  # [1,1]などの組み合わせを削除
        if edges_indices.shape[0] == 0:
            continue
        edges_indices = np.sort(edges_indices, axis=1)
        edges_indices = np.unique(edges_indices, axis=0)  # 重複を削除

        while edges_indices.shape[0] == 0:
            edges_indices = np.random.randint(0, node_num, (edge_num, 2))
            edges_indices = edges_indices[edges_indices[:, 0] != edges_indices[:, 1]]
            edges_indices = np.sort(edges_indices, axis=1)
            edges_indices = np.unique(edges_indices, axis=0)
        ref_edges_indices = edges_indices.copy()

        exist_nodes = np.unique(edges_indices)
        node_num = exist_nodes.shape[0]
        if node_num >= 2:  # 2こ以上無いと固定ノード，荷重ノードのどちらかが0になる
            flag = 1
        else:
            continue
        edge_num = edges_indices.shape[0]
        ref_nodes = np.arange(node_num)

        for exist_node, ref_node in zip(exist_nodes, ref_nodes):
            edges_indices[ref_edges_indices == exist_node] = ref_node

        # ノードの位置およびエッジの太さをランダム設定
        nodes_pos = np.random.uniform(0, 1, (node_num, 2))
        edges_thickness = np.random.uniform(0.1, 1, edge_num)

        # 固定ノード，外力ノードを選択
        frozen_num = np.random.randint(1, node_num)
        frozen_nodes = np.unique(np.random.randint(0, node_num, frozen_num)).tolist()
        input_nodes = np.random.randint(0, node_num, 1).tolist()
        while input_nodes[0] in frozen_nodes:
            input_nodes = np.random.randint(0, node_num, 1).tolist()
        input_vectors = np.random.rand(1, 2)

    return nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes


def make_random_fem_condition_with_ER(max_node_num, max_edge_pos):
    """ランダムにFEMの条件をERグラフを利用して生成する

    Args:
        max_node_num (int): 最大のノードの数
        max_edge_num (int): 最大のエッジの数
    """
    assert max_node_num > 4, "max_node_num should be bigger than 4"

    while True:
        node_num = np.random.randint(4, max_node_num)
        edge_possibility = np.random.uniform(0.3, max_edge_pos)
        G = nx.fast_gnp_random_graph(node_num, edge_possibility)
        if nx.is_connected(G):
            break
    edge_num = nx.number_of_edges(G)  # エッジの数の再取得
    edges_indices = np.array(G.edges())
    # ノードの位置およびエッジの太さをランダム設定
    nodes_pos = np.random.uniform(0, 1, (node_num, 2))
    edges_thickness = np.random.uniform(0.01, 1, edge_num)

    # 固定ノード，外力ノードを選択
    frozen_num = np.random.randint(1, int(node_num / 2))
    frozen_nodes = np.unique(np.random.randint(0, node_num, frozen_num)).tolist()
    input_nodes = np.random.randint(0, node_num, 1).tolist()
    while input_nodes[0] in frozen_nodes:
        input_nodes = np.random.randint(0, node_num, 1).tolist()
    input_vectors = np.random.rand(1, 2)

    return nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes
