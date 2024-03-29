import numpy as np


def make_main_node_edge_info(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                             origin_output_nodes, origin_output_vectors, origin_frozen_nodes, condition_edge_thickness=0.2):
    """envが処理できるように，元のnode_posやedge_indiceから必要な情報のみを取り出してくる関数．
    出力されるnode_posの順番は入力ノード，出力ノード，固定ノードの順番，そして，ノード番号はそれぞれ0,1,2...に直される．
    そして全てのエッジの太さは condition_edge_thicknessとして出力される．

    Args:
        origin_nodes_positions (np.array): 情報を抽出したいnode_pos
        origin_edges_indices (np.array): 情報を抽出したいedge_indices
        origin_input_nodes (list): 抽出したい入力ノードの番号
        origin_input_vectors (np.array): 入力ノードに対する入力方向,[n,2]
        origin_output_nodes (list): 抽出したい出力ノードの番号
        origin_output_vectors (np.array): 出力ノードに対する出力方向,[n,2]
        origin_frozen_nodes (list): 抽出したい固定ノードの番号
        condition_edge_thickness (float, optional): 初期のエッジの太さ. Defaults to 0.2.
    """

    new_input_nodes = np.arange(len(origin_input_nodes))
    new_output_nodes = np.arange(
        len(origin_output_nodes)) + len(origin_input_nodes)
    new_frozen_nodes = np.arange(
        len(origin_frozen_nodes)) + len(origin_input_nodes) + len(origin_output_nodes)

    if origin_frozen_nodes != []:
        new_node_pos = origin_nodes_positions[np.concatenate([origin_input_nodes, origin_output_nodes, origin_frozen_nodes])]
    else:
        new_node_pos = origin_nodes_positions[np.concatenate([origin_input_nodes, origin_output_nodes])]

    new_edges_indices = make_main_edges_indices(
        origin_edges_indices, origin_input_nodes, origin_output_nodes, origin_frozen_nodes)

    new_edges_thickness = np.ones(
        len(new_edges_indices)) * condition_edge_thickness
    assert len(origin_input_nodes) == origin_input_vectors.shape[0], "the number of origin_input_nodes should be same as input_vectors"
    assert len(origin_output_nodes) == origin_output_vectors.shape[0], "the number of origin_output_nodes should be same as output_vectors"
    new_input_vectors = origin_input_vectors
    new_output_vectors = origin_output_vectors

    return new_node_pos, new_input_nodes.tolist(), new_input_vectors, new_output_nodes.tolist(), new_output_vectors, new_frozen_nodes.tolist(), new_edges_indices, new_edges_thickness


def make_main_edges_indices(edges_indices, input_nodes, output_nodes, frozen_nodes):
    """初期条件ノードの番号を基に，edges_indicesのうちに初期条件ノードのみを含むものを，ノード番号を変えて
    新しいedges_indicesを作成する．
    また，新しいノード番号は入力ノード，出力ノード，固定ノードの順に0,1,2,3となる．
    """
    if frozen_nodes != []:
        valid_nodes = np.concatenate([input_nodes, output_nodes, frozen_nodes])
    else:
        valid_nodes = np.concatenate([input_nodes, output_nodes])
    # edges_indicesのうち，初期エッジ情報のみを抽出する
    edges_bool = np.isin(edges_indices, valid_nodes)
    edges_bool = edges_bool[:, 0] & edges_bool[:, 1]
    edges_indices = edges_indices[edges_bool]

    new_edges_indices = np.zeros(shape=edges_indices.shape, dtype=int)
    for i, v in enumerate(valid_nodes):
        mask = edges_indices == v
        new_edges_indices[mask] = i

    return new_edges_indices


def make_continuous_init_graph(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                               origin_output_nodes, origin_output_vectors, origin_frozen_nodes, condition_edge_thickness):
    """連続状態をランダムで生成する．

    Args:
        origin_nodes_positions (np.array): 
        origin_edges_indices (np.array): 
        origin_input_node (list)): 
        origin_input_vectors (np.array): 
        origin_output_nodes (list)): 
        origin_output_vectors (np.array): 
        origin_frozen_nodes (list)): 
        condition_edge_thickness (float): 環境エッジの太さ
    """

    new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness = make_main_node_edge_info(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                                                                                                                                                                                origin_output_nodes, origin_output_vectors, origin_frozen_nodes, condition_edge_thickness)

    # 固定ノード部分同士の連結しか情報を持たないことを想定
    assert (not np.isin(0, new_edges_indices)) or (not np.isin(
        1, new_edges_indices)), "origin_edge_indices includes input and output node's indices"

    rand_edges_thickness = np.random.rand(8)
    rand_node_pos = np.random.rand(6, 2)

    new_node_pos = np.concatenate([new_node_pos, rand_node_pos])
    new_edges_thickness = np.concatenate(
        [new_edges_thickness, rand_edges_thickness])

    frozen_node_num = len(new_frozen_nodes)
    input_add_edge_indices = [[0, frozen_node_num + 2], [frozen_node_num + 2, frozen_node_num + 3], [
        frozen_node_num + 3, frozen_node_num + 4], [frozen_node_num + 4, np.random.randint(2, frozen_node_num + 2)]]
    output_add_edge_indices = [[1, frozen_node_num + 5], [frozen_node_num + 5, frozen_node_num + 6], [
        frozen_node_num + 6, frozen_node_num + 7], [frozen_node_num + 7, np.random.randint(2, frozen_node_num + 2)]]

    new_edges_indices = np.concatenate(
        [new_edges_indices, input_add_edge_indices, output_add_edge_indices])

    return new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness
