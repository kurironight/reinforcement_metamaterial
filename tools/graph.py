import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def convert_edge_indices_to_adj(edges_indices, size=False):
    """
    隣接行列を作成する．sizeが指定されているとき，size*sizeの隣接行列にして出力する．

    Args:
        edges_indices ([type]): node_num*2

    return:
        node_num*node_num or size*size
    """
    node_num = edges_indices.max() + 1
    node_adj = np.zeros((node_num, node_num), dtype=int)
    node_adj[(edges_indices[:, 0], edges_indices[:, 1])] = 1
    node_adj[(edges_indices[:, 1], edges_indices[:, 0])] = 1

    if size:
        node_adj = np.pad(
            node_adj, ((0, size - node_num), (0, size - node_num)))

    return node_adj


def convert_adj_to_edge_indices(adj):
    """
    隣接行列からエッジのindex情報を作成する

    Args:
        adj (np.array): node_num*node_num

    return:
        edge_num*2
    """
    index = np.stack([np.where(adj)[0], np.where(adj)[1]]).T

    # 重複をなくす
    mask = index[:, 0] < index[:, 1]
    edge_indices = index[mask]

    return edge_indices


def make_T_matrix(edges_indices):
    """T行列を作成する関数

    Args:
        edges_indices (np.array): edge_num*2

    Returns:
        T(np.array): node_num*edge_num
    """
    node_num = edges_indices.max() + 1
    edge_num = edges_indices.shape[0]
    T = np.zeros((edge_num, node_num), dtype=np.int32)
    for i, edge_indice in enumerate(edges_indices):
        T[i][edge_indice] = 1
    T = T.T
    return T


def make_edge_adj(edges_indices, T):
    """エッジの隣接行列を作成する．

    Args:
        edges_indices (np.array): edge_num*2
        T (np.array): node_num*edge_num

    Returns:
        edge_adj (np.array): edge_num*edge_num
    """
    edge_num = edges_indices.shape[0]
    edge_adj = np.zeros((edge_num, edge_num), dtype=np.int32)
    for i, edge_indice in enumerate(edges_indices):
        # i番目のエッジがどのノードと繋がっているか抽出
        node1 = edge_indice[0]
        node2 = edge_indice[1]
        # そのノードがどのエッジと繋がっているか抽出
        connected_edges1 = np.where(T[node1])
        connected_edges2 = np.where(T[node2])
        # 重複を除く
        connect_edges = np.unique(np.concatenate(
            [connected_edges1, connected_edges2], axis=1))

        edge_adj[i][connect_edges] = 1

    # 最後に対角成分を0にする
    np.fill_diagonal(edge_adj, 0)

    return edge_adj


def make_D_matrix(adj):
    """隣接行列adj
    I+adjの対角次数行列Dを作成する関数

    Args:
        adj (np.array): n*n

    Returns:
        D (np.array): n*n
    """
    D = np.diag(np.sum(adj, axis=0) + 1)  # +1は単位行列Iを考慮したもの
    return D


def calc_cross_point(pointA, pointB, pointC, pointD):
    """二つの線分の交差点を求める関数．交差している場合，True．交差していない場合，Falseを出力する．

    Args:
        pointA (np.array): 線分１の始点
        pointB (np.array): 線分１の終点
        pointC (np.array): 線分２の始点
        pointD (np.array): 線分２の終点

    Returns:
        [bool,np.array]: 交差しているか，そしてその交点．交差していない場合，[0,0]を返す．
    """
    cross_point = np.array([0.0, 0.0])
    bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

    # 直線が平行な場合
    if (bunbo == 0):
        return False, cross_point

    vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
    r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
    s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo

    if (0 < r and r < 1) and (0 < s and s < 1):  # 交差しているとき TODO 片方のエッジ上にノードが乗っている場合，s=0 ors=1 もしくは逆
        # rを使った計算の場合
        distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
        cross_point = np.array([pointA[0] + distance[0], pointA[1] + distance[1]])
        return True, cross_point
    else:
        return False, cross_point


def calc_corresp_line(pointA, pointB, pointC, pointD):
    """二つの線分が直線的に一致しているかを求める関数．一致している場合，True．一致していない場合，Falseを出力する．

    Args:
        pointA (np.array): 線分１の始点
        pointB (np.array): 線分１の終点
        pointC (np.array): 線分２の始点
        pointD (np.array): 線分２の終点

    Returns:
        [bool,np.array]: 一致しているか，一致していないかを判定する．
    """
    bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

    vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
    r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1])  # 分母
    s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1])  # 分母

    # 直線が平行な場合
    if (bunbo == 0 and r == 0 and s == 0):
        return True
    else:
        return False


def separate_same_line_group(nodes_pos, edges_indices):
    # 同じ直線上にあるエッジをリストごとにまとめる．そして，独立しているエッジもリストにまとめる．
    same_line_groups = []
    independent_group = []
    edge_points = np.array([[nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]] for edges_indice in edges_indices])
    while edges_indices.shape[0] != 0:
        ref_same_line_group = []
        ref_edge_indice = edges_indices[0]
        ref_edge_point = edge_points[0]
        ref_same_line_group.append(ref_edge_indice)
        edges_indices = np.delete(edges_indices, 0, 0)
        edge_points = np.delete(edge_points, 0, 0)
        for edge_point, edges_indice in zip(edge_points.copy(), edges_indices.copy()):
            if calc_corresp_line(ref_edge_point[0], ref_edge_point[1], edge_point[0], edge_point[1]):
                ref_same_line_group.append(edges_indice)
                remove_index = find_edge_indice_index(edges_indice, edges_indices)
                edge_points = np.delete(edge_points, remove_index, axis=0)
                edges_indices = np.delete(edges_indices, remove_index, axis=0)

        if len(ref_same_line_group) != 1:
            same_line_groups.append(ref_same_line_group)
        else:
            independent_group.append(ref_same_line_group)

    same_line_groups = np.array(same_line_groups) if len(same_line_groups) != 0 else []
    independent_group = np.squeeze(np.array(independent_group), 1) if len(independent_group) != 0 else []

    return same_line_groups, independent_group


def make_same_line_group_edge(nodes_pos, edges_indices, edges_thickness):
    # 同じ位置のノードを別々に扱わないことを前提にしている
    assert np.unique(nodes_pos, axis=0).shape[0] == nodes_pos.shape[0], "there should be no same nodes pos with different node number"

    def edge1_max_edge2_max_indices():
        return np.sort([edges_indices_combination[edge1_index][np.argmax(edge1)], edges_indices_combination[edge2_index][np.argmax(edge2)]])

    def edge1_min_edge2_min_indices():
        return np.sort([edges_indices_combination[edge1_index][np.argmin(edge1)], edges_indices_combination[edge2_index][np.argmin(edge2)]])

    def edge1_max_edge2_min_indices():
        return np.sort([edges_indices_combination[edge1_index][np.argmax(edge1)], edges_indices_combination[edge2_index][np.argmin(edge2)]])

    def search_refer_edge_thickness(target_edges_indices, refer_edges_indices, refer_edge_thickness):
        return refer_edge_thickness[(refer_edges_indices[:, 0] == target_edges_indices[0]) & (refer_edges_indices[:, 1] == target_edges_indices[1])].squeeze()

    trigger = 1
    while trigger == 1:
        edge_points = [np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices]
        combinations = [pair for pair in itertools.combinations(edge_points, 2)]
        edges_indices_combinations = [pair for pair in itertools.combinations(edges_indices, 2)]
        new_edges_indices = []
        new_edges_thickness = []
        erased_edge_indices = []

        for combination, edges_indices_combination in zip(combinations, edges_indices_combinations):
            # 通常はx座標のみでエッジの重なりパターンを判断する
            refer_index = 0 if combination[0][0][0] != combination[0][1][0] else 1
            # 最小の参考値を持っているエッジをエッジ1とする
            edge1_index = 0 if min(combination[0][:, refer_index]) <= min(combination[1][:, refer_index]) else 1
            edge2_index = 1 - edge1_index
            edge1 = combination[edge1_index][:, refer_index]
            edge2 = combination[edge2_index][:, refer_index]

            if max(edge1) < min(edge2):  # pattern1
                # print("pattern1")
                new_edges_indices.append(edges_indices_combination[0])
                new_edges_indices.append(edges_indices_combination[1])
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[0], edges_indices, edges_thickness))
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[1], edges_indices, edges_thickness))
            elif max(edge1) == min(edge2):  # pattern5
                # print("pattern5")
                new_edges_indices.append(edges_indices_combination[edge1_index])
                new_edges_indices.append(edge1_max_edge2_max_indices())
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness))
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness))
            elif max(edge1) == max(edge2) and min(edge1) != min(edge2):  # pattern6
                # print("pattern6")
                new_edges_indices.append(edge1_min_edge2_min_indices())
                new_edges_indices.append(edge1_max_edge2_min_indices())
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness))
                new_edges_thickness.append(max([search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness),
                                                search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness)]))
                erased_edge_indices.append(edges_indices_combination[edge1_index])
            elif max(edge1) != max(edge2) and min(edge1) == min(edge2):  # pattern7
                # redefine edge1 and edge2 rely to the max num
                edge1_index = 0 if max(edge1) < max(edge2) else 1
                edge2_index = 1 - edge1_index
                edge1 = combination[edge1_index][:, refer_index]
                edge2 = combination[edge2_index][:, refer_index]
                # print("pattern7")
                new_edges_indices.append(edges_indices_combination[edge1_index])
                new_edges_indices.append(edge1_max_edge2_max_indices())
                new_edges_thickness.append(max([search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness),
                                                search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness)]))
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness))
                erased_edge_indices.append(edges_indices_combination[edge2_index])
            elif max(edge2) > max(edge1) and max(edge1) > min(edge2):  # pattern2
                # print("pattern2")
                new_edges_indices.append(edge1_min_edge2_min_indices())
                new_edges_indices.append(edge1_max_edge2_min_indices())
                new_edges_indices.append(edge1_max_edge2_max_indices())
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness))
                new_edges_thickness.append(max([search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness),
                                                search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness)]))
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness))
                erased_edge_indices.append(edges_indices_combination[0])
                erased_edge_indices.append(edges_indices_combination[1])
            elif max(edge1) > max(edge2):  # pattern3
                # print("pattern3")
                new_edges_indices.append(edge1_min_edge2_min_indices())
                new_edges_indices.append(edges_indices_combination[edge2_index])
                new_edges_indices.append(edge1_max_edge2_max_indices())
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness))
                new_edges_thickness.append(max([search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness),
                                                search_refer_edge_thickness(edges_indices_combination[edge2_index], edges_indices, edges_thickness)]))
                new_edges_thickness.append(search_refer_edge_thickness(edges_indices_combination[edge1_index], edges_indices, edges_thickness))
                erased_edge_indices.append(edges_indices_combination[edge1_index])
        refer_new_edges_indices = np.array(new_edges_indices).copy()  # ソート済
        new_edges_indices = np.unique(np.array(new_edges_indices), axis=0)

        if erased_edge_indices != []:  # 除外するリストを用意し，除去する．
            erased_edge_indices = np.sort(np.array(erased_edge_indices), axis=1)
            erased_edge_indices = np.unique(erased_edge_indices, axis=0)

            for i in erased_edge_indices:
                new_edges_indices = np.array([j for j in new_edges_indices if not np.array_equal(i, j)])

        if np.array_equal(edges_indices, new_edges_indices):
            trigger = 0
        edges_indices = new_edges_indices

        new_edges_thickness = np.array(new_edges_thickness)
        edges_thickness = []
        for target_edge_indice in new_edges_indices:
            edges_thickness.append(np.max(new_edges_thickness[find_edge_indice_index(target_edge_indice, refer_new_edges_indices)]))
        edges_thickness = np.array(edges_thickness)

    return edges_indices, edges_thickness


def find_edge_indice_index(target_edges_indice, ref_edge_indices):
    return np.argwhere((ref_edge_indices[:, 0] == target_edges_indice[0]) & (ref_edge_indices[:, 1] == target_edges_indice[1])).squeeze()


def find_nodes_pos_index(target_node_pos, ref_nodes_pos):
    return np.argwhere((ref_nodes_pos[:, 0] == target_node_pos[0]) & (ref_nodes_pos[:, 1] == target_node_pos[1])).squeeze()


def separate_same_line_procedure(nodes_pos, edges_indices, edges_thickness):
    """同じ直線上に存在するエッジを考慮し，エッジ分割を行う

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
        edges_thickness (np.array): (*)
    """
    sl_groups, independent_group = separate_same_line_group(nodes_pos, edges_indices)

    revised_edges_indices = np.empty((0, 2), int)
    revised_edges_thickness = np.empty(0)

    if sl_groups != []:
        for sl_group_edge_indices in sl_groups:
            sl_group_edge_indices = np.array(sl_group_edge_indices).reshape((-1, 2))
            sl_group_edges_thickness = np.array([edges_thickness[find_edge_indice_index(target_edges_indice, edges_indices)] for target_edges_indice in sl_group_edge_indices])
            revised_sl_group_edge_indices, revised_sl_edges_thickness = \
                make_same_line_group_edge(nodes_pos, sl_group_edge_indices, sl_group_edges_thickness)
            revised_edges_indices = np.append(revised_edges_indices, revised_sl_group_edge_indices, axis=0)
            revised_edges_thickness = np.append(revised_edges_thickness, revised_sl_edges_thickness, axis=0)

    if independent_group != []:
        id_group_edges_thickness = np.array([edges_thickness[find_edge_indice_index(target_edges_indice, edges_indices)] for target_edges_indice in independent_group])
        revised_edges_indices = np.append(revised_edges_indices, independent_group, axis=0)
        revised_edges_thickness = np.append(revised_edges_thickness, id_group_edges_thickness, axis=0)

    return revised_edges_indices, revised_edges_thickness


def preprocess_graph_info(nodes_pos, edges_indices, edges_thickness):
    """同じ座標を示すノードの排除，[1,1]などのエッジの排除，エッジの[2,1]を[1,2]にするなどのソートなどを行う

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
        edges_thickness (np.array): (*)
    """
    # edge_indicesをaxis1方向にソートする．
    edges_indices = np.sort(edges_indices, axis=1)
    # 同じ位置にあるノードのedge_indiceにおける番号を統一する
    remove_node_index = np.empty(0, int)
    for node_index, node_pos in enumerate(nodes_pos):
        same_index = np.argwhere([np.allclose(node_pos, ref_node_pos) for ref_node_pos in nodes_pos])
        if same_index.shape[0] != 1:
            ident_node_index = min(same_index)
            erased_node_index = np.setdiff1d(same_index, ident_node_index)
            edges_indices[np.isin(edges_indices, erased_node_index)] = ident_node_index
            remove_node_index = np.append(remove_node_index, erased_node_index, axis=0)
    # 抜けたノードの分，ノードのindexをedge_indicesに指定しなおす
    ref_nodes_pos = nodes_pos.copy()
    remove_node_index = np.unique(remove_node_index)
    processed_nodes_pos = np.delete(nodes_pos, remove_node_index, 0)  # 被りがないnode_pos
    processed_edges_indices = edges_indices.copy()
    for i, target_node_pos in enumerate(processed_nodes_pos):
        processed_edges_indices[edges_indices == np.min(find_nodes_pos_index(target_node_pos, ref_nodes_pos))] = i
    ref_processed_edges_indices = processed_edges_indices.copy()

    # edge_indiceの内，被りがあるものを除去する
    processed_edges_indices = np.unique(np.array(processed_edges_indices), axis=0)

    # 除去したもののedge_thickを除去する
    processed_edges_thickness = []
    for target_edge_indice in processed_edges_indices:
        processed_edges_thickness.append(np.max(edges_thickness[find_edge_indice_index(target_edge_indice, ref_processed_edges_indices)]))
    processed_edges_thickness = np.array(processed_edges_thickness)

    # edge_indicesのうち，[1,1]などのように同じノードを示しているものを除去する
    ident_edge_index = np.argwhere(processed_edges_indices[:, 0] == processed_edges_indices[:, 1])
    if ident_edge_index.shape[0] != 0:
        processed_edges_indices = np.delete(processed_edges_indices, ident_edge_index, 0)
        processed_edges_thickness = np.delete(processed_edges_thickness, ident_edge_index, 0)

    return processed_nodes_pos, processed_edges_indices, processed_edges_thickness


def conprocess_seperate_edge_indice_procedure(condition_input_nodes, condition_output_nodes, condition_frozen_nodes, condition_nodes_pos, condition_edges_indices, condition_edges_thickness,
                                              processed_nodes_pos, processed_edges_indices, processed_edges_thickness):
    """入力ノード，出力ノード，固定ノードのindexを指定しなおす．また，条件ノード間のエッジの太さを元に戻す．
    """
    # TODO 条件ノード間に新しいノードが追加された場合，エッジの太さを基に戻すことが出来なくなっている恐れがある．
    # input_nodesのindexを再指定
    input_nodes = [find_nodes_pos_index(condition_nodes_pos[target_node], processed_nodes_pos) for target_node in condition_input_nodes]
    # output_nodesのindexを再指定
    output_nodes = [find_nodes_pos_index(condition_nodes_pos[target_node], processed_nodes_pos) for target_node in condition_output_nodes]
    # frozen_nodesのindexを再指定
    frozen_nodes = [find_nodes_pos_index(condition_nodes_pos[target_node], processed_nodes_pos) for target_node in condition_frozen_nodes]
    # processed_edges_thicknessの固定部分をcondition_edge_thicknessに再指定
    condition_edge_points = np.array([[condition_nodes_pos[edges_indice[0]], condition_nodes_pos[edges_indice[1]]] for edges_indice in condition_edges_indices])
    processed_edge_points = np.array([[processed_nodes_pos[edges_indice[0]], processed_nodes_pos[edges_indice[1]]] for edges_indice in processed_edges_indices])
    for i, t in zip(condition_edge_points, condition_edges_thickness):
        target_index = np.argwhere(np.array([np.array_equal(j, i[0]) for j in processed_edge_points[:, 0]]) & np.array([np.array_equal(j, i[1]) for j in processed_edge_points[:, 1]]))
        if target_index.shape[0] == 0:
            target_index = np.argwhere(np.array([np.array_equal(j, i[1]) for j in processed_edge_points[:, 0]]) & np.array([np.array_equal(j, i[0]) for j in processed_edge_points[:, 1]]))
        processed_edges_thickness[target_index] = t

    return input_nodes, output_nodes, frozen_nodes, processed_edges_thickness


def check_cross_graph(nodes_pos, edges_indices):
    """グラフ内に交差しているエッジがあるかどうかをチェックする

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
    """
    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    edge_points_combinations = [pair for pair in itertools.combinations(edge_points, 2)]
    for edge_points in edge_points_combinations:
        cross, cross_point = calc_cross_point(edge_points[0][0], edge_points[0][1], edge_points[1][0], edge_points[1][1])
        if cross:  # 交差しているとき
            return True

    return False


def seperate_cross_line_procedure(nodes_pos, edges_indices, edges_thickness):
    """交差しているエッジについて，交差点を新しいノードにし，エッジ分割を行う

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
        edges_thickness (np.array): (*)
    """
    edge_info = np.concatenate([edges_indices, edges_thickness.reshape(-1, 1)], axis=1)  # edge_indiceとedge_thicknessを結合した

    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    edge_index_combinations = [pair for pair in itertools.combinations(np.arange(len(edge_points)), 2)]
    edge_points_combinations = [pair for pair in itertools.combinations(edge_points, 2)]
    each_edge_add_points = dict([(i, edges_indices[i].tolist()) for i in range(len(edge_points))])
    for edge_index, edge_points in zip(edge_index_combinations, edge_points_combinations):
        cross, cross_point = calc_cross_point(edge_points[0][0], edge_points[0][1], edge_points[1][0], edge_points[1][1])
        if cross:  # 交差しているとき
            nodes_pos = np.vstack((nodes_pos, cross_point))  # ノード追加
            each_edge_add_points[edge_index[0]].append(len(nodes_pos) - 1)  # 各エッジに追加する予定のノードのindexをdictに追加している
            each_edge_add_points[edge_index[1]].append(len(nodes_pos) - 1)  # 各エッジに追加する予定のノードのindexをdictに追加している
    remove_edge_index = []
    # 一直線上に存在するnode_posをグループ分けし，x座標順に並べる
    for edge_index, node_index_group in each_edge_add_points.items():
        if len(node_index_group) != 2:  # 分割されたエッジを示している場合
            edge_thickness = edges_thickness[edge_index]
            # x座標でソートする場合，refer_index=0.y座標の時，1.
            refer_index = 0 if nodes_pos[edges_indices[edge_index][0]][0] != nodes_pos[edges_indices[edge_index][1]][0] else 1
            index_order = np.argsort([nodes_pos[node_index][refer_index] for node_index in node_index_group])
            node_index_order_group = np.array(node_index_group)[index_order]  # 座標順にソートされたノードのインデックス
            unjointed_edge_info = np.array([[node_index_order_group[i], node_index_order_group[i + 1], edge_thickness] for i in range(len(node_index_order_group) - 1)])
            edge_info = np.append(edge_info, unjointed_edge_info, axis=0)  # 分割されたエッジを追加する

            remove_edge_index.append(edge_index)

    # 後処理で必要なのは，ソート，重複取り消し，同じノード部分の取り消し
    edge_info = np.delete(edge_info, remove_edge_index, 0)

    processed_edges_indices = edge_info[:, :2].astype(int)
    processed_edges_thickness = edge_info[:, 2]
    processed_nodes_pos = nodes_pos

    return processed_nodes_pos, processed_edges_indices, processed_edges_thickness


def remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, nodes_pos, edges_indices):
    """edges_indicesで触れられていないノードがnodes_posに存在する時，これらを除外し，
    かつノードのindexを再振り分けした上でedges_indicesやinput_nodesなどを返す．

    Args:
        input_nodes (list):
        output_nodes (list):
        frozen_nodes (list):
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
    """
    input_nodes = np.array(input_nodes)
    frozen_nodes = np.array(frozen_nodes)
    output_nodes = np.array(output_nodes)
    node_num = nodes_pos.shape[0]
    mask = np.isin(np.arange(node_num), edges_indices)
    if not np.all(mask):  # edges_indicesで触れられていないノードがnodes_posに存在する時，これらを除外したうえで，barfemにかける
        processed_input_nodes = input_nodes.copy()
        processed_frozen_nodes = frozen_nodes.copy()
        processed_output_nodes = output_nodes.copy()
        processed_edges_indices = edges_indices.copy()
        prior_index = np.arange(node_num)[mask]
        processed_nodes_pos = nodes_pos[mask]
        for index, prior_index in enumerate(prior_index):
            if index != prior_index:
                processed_edges_indices[edges_indices ==
                                        prior_index] = index
                # input_nodesとfrozen_nodes部分のラベルを変更
                processed_input_nodes[input_nodes == prior_index] = index
                processed_output_nodes[output_nodes == prior_index] = index
                processed_frozen_nodes[frozen_nodes == prior_index] = index
        nodes_pos = processed_nodes_pos
        edges_indices = processed_edges_indices
        input_nodes = processed_input_nodes
        output_nodes = processed_output_nodes
        frozen_nodes = processed_frozen_nodes
    input_nodes = input_nodes.tolist()
    output_nodes = output_nodes.tolist()
    frozen_nodes = frozen_nodes.tolist()

    return input_nodes, output_nodes, frozen_nodes, nodes_pos, edges_indices


def calc_efficiency(input_nodes, input_vectors, output_nodes, output_vectors, displacement):
    denominator = np.sum([np.dot(input_vectors[i] / np.linalg.norm(input_vectors[i]), displacement[[input_node * 3 + 0, input_node * 3 + 1]]) for i, input_node in enumerate(input_nodes)])
    efficiency = np.dot(output_vectors / np.linalg.norm(output_vectors), displacement[[output_nodes[0] * 3 + 0, output_nodes[0] * 3 + 1]]) / denominator
    return efficiency


def render_graph(nodes_pos, edges_indices, edges_thickness, save_path, display_number=False):
    """グラフを図示
    Args:
        save_path (str, optional): 図を保存するパス.
        display_number (bool, optional): ノードに番号をつけるか付けないか. Defaults to False.
    """
    edge_size = 30  # 図示する時のエッジの太さ
    marker_size = 40  # 図示するときのノードのサイズ
    character_size = 20  # ノードの文字のサイズ

    starts = nodes_pos[edges_indices[:, 0]]
    ends = nodes_pos[edges_indices[:, 1]]

    lines = [(start, end) for start, end in zip(starts, ends)]

    lines = LineCollection(lines, linewidths=edges_thickness * edge_size)

    plt.clf()  # Matplotlib内の図全体をクリアする
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], s=marker_size, c="red", zorder=2)
    if display_number:
        for i, txt in enumerate(["{}".format(i) for i in range(nodes_pos.shape[0])]):
            ax.annotate(txt, (nodes_pos[i, 0], nodes_pos[i, 1]), size=character_size, horizontalalignment="center", verticalalignment="center")
    ax.autoscale()
    plt.savefig(save_path)
    plt.close()
