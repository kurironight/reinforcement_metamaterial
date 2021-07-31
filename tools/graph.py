import numpy as np
import itertools


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

    if (0 < r and r < 1) and (0 < s and s < 1):
        # rを使った計算の場合
        distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
        cross_point = (pointA[0] + distance[0], pointA[1] + distance[1])
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
        for edge_indice in new_edges_indices:
            edges_thickness.append(np.max(new_edges_thickness[(refer_new_edges_indices[:, 0] == edge_indice[0]) & (refer_new_edges_indices[:, 1] == edge_indice[1])]))
        edges_thickness = np.array(edges_thickness)

    return edges_indices, edges_thickness


def find_edge_indice_index(target_edges_indice, ref_edge_indices):
    return np.argwhere((ref_edge_indices[:, 0] == target_edges_indice[0]) & (ref_edge_indices[:, 1] == target_edges_indice[1])).squeeze()
