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


def make_random_fem_condition_with_continuous_graph(max_node_num, max_edge_pos):
    node_num = np.random.randint(2, max_node_num)
    one_side_node_num = int(node_num / 2)  # 片方の塊のノード数

    # 各ノードの配置
    frozen_nodes_pos = np.array([[0, 0], [1 / 3, 0], [2 / 3, 0], [1, 0]])
    frozen_node_num = frozen_nodes_pos.shape[0]
    input_node_pos = np.array([[1, 1]])  # node_num;[frozen_node_num]
    output_node_pos = np.array([[0, 1]])  # node_num;[frozen_node_num+1]
    input_rand_node_pos = np.random.uniform(0, 1, (one_side_node_num, 2))  # node_num;[frozen_node_num+2,..,frozen_node_num+2+one_side_node_num-1]
    output_rand_node_pos = np.random.uniform(0, 1, (one_side_node_num, 2))  # node_num;[frozen_node_num+2+one_side_node_num,..,frozen_node_num+2+2*one_side_node_num-1]

    nodes_pos = np.concatenate([frozen_nodes_pos, input_node_pos, output_node_pos, input_rand_node_pos, output_rand_node_pos])

    # edge_indiceの設定
    frozen_edge_indices = np.array([[0, 1], [1, 2], [2, 3]])
    first_input_node = frozen_node_num + 2
    first_output_node = frozen_node_num + 2 + one_side_node_num
    input_add_edge_indices = [[frozen_node_num, first_input_node]]
    input_add_edge_indices.extend([[first_input_node + i, first_input_node + 1 + i] for i in range(one_side_node_num - 1)])
    input_add_edge_indices.extend([[first_input_node + one_side_node_num - 1, np.random.randint(0, frozen_node_num)]])  # 固定ノード部分の結合

    output_add_edge_indices = [[frozen_node_num + 1, first_output_node]]
    output_add_edge_indices.extend([[first_output_node + i, first_output_node + 1 + i] for i in range(one_side_node_num - 1)])
    output_add_edge_indices.extend([[first_output_node + one_side_node_num - 1, np.random.randint(0, frozen_node_num)]])  # 固定ノード部分の結合
    # input部分とoutput部分の交差部分のエッジを追加
    edge_possibility = np.random.uniform(0.3, max_edge_pos)
    dice = [1, 0]
    prob = [edge_possibility, 1 - edge_possibility]
    cross_edge_indices = np.random.choice(a=dice, size=(one_side_node_num, one_side_node_num), p=prob)
    cross_edge_indices = np.array(list(zip(*np.where(cross_edge_indices == 1))))
    if cross_edge_indices != np.array([]):
        cross_edge_indices[:, 1] += one_side_node_num
        cross_edge_indices += frozen_node_num + 2
        edges_indices = np.concatenate(
            [frozen_edge_indices, input_add_edge_indices, output_add_edge_indices, cross_edge_indices])
    else:
        edges_indices = np.concatenate(
            [frozen_edge_indices, input_add_edge_indices, output_add_edge_indices])
    assert np.unique(edges_indices, axis=0).shape[0] == edges_indices.shape[0], "どっかで間違っている"

    edge_num = edges_indices.shape[0]
    random_edges_thickness = np.random.uniform(0.01, 1, edge_num - frozen_edge_indices.shape[0])
    edges_thickness = np.concatenate([np.array([1, 1, 1]), random_edges_thickness])

    # 固定ノード，外力ノードを選択
    frozen_nodes = list(range(frozen_node_num))
    input_nodes = [first_input_node]
    input_vectors = np.array([[0, -1]])

    return nodes_pos, edges_indices, edges_thickness, input_nodes, input_vectors, frozen_nodes
