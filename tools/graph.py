import networkx as nx
from networkx.classes.function import edges
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from FEM.make_structure import make_bar_structure


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
    # target_node_posがref_node_posのどこにあるのかを求める関数．
    # 複数同じnode_posが存在する場合，一番indexの小さいものを出力する．
    return np.min(np.argwhere((ref_nodes_pos[:, 0] == target_node_pos[0]) & (ref_nodes_pos[:, 1] == target_node_pos[1])))


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
    # 同じ位置にあるノードのedge_indiceにおける番号を統一する
    remove_node_index = np.empty(0, int)
    indexes = np.arange(nodes_pos.shape[0]).reshape((nodes_pos.shape[0], 1))
    nodes_pos_info = np.concatenate([nodes_pos, indexes], axis=1)
    while True:
        ref_node_pos_info = nodes_pos_info[0]
        ref_node_pos = ref_node_pos_info[:2]
        ref_node_index = ref_node_pos_info[2]
        nodes_pos_info = np.delete(nodes_pos_info, 0, 0)
        same_index = np.argwhere([(np.allclose(node_pos, ref_node_pos) & np.allclose(ref_node_pos, node_pos)) for node_pos in nodes_pos_info[:, :2]])
        if same_index.shape[0] != 0:
            erased_node_index = nodes_pos_info[:, 2][same_index]
            edges_indices[np.isin(edges_indices, erased_node_index)] = ref_node_index
            remove_node_index = np.append(remove_node_index, erased_node_index.reshape((-1)), axis=0)
            nodes_pos_info = np.delete(nodes_pos_info, same_index, 0)
        if nodes_pos_info.shape[0] == 0:
            break
    ref_nodes_pos = nodes_pos.copy()
    processed_nodes_pos = np.delete(nodes_pos, remove_node_index, 0)  # 被りがないnode_pos
    processed_edges_indices = edges_indices.copy()
    if remove_node_index.shape[0] != 0:
        for i, target_node_pos in enumerate(processed_nodes_pos):
            processed_edges_indices[edges_indices == find_nodes_pos_index(target_node_pos, ref_nodes_pos)] = i

    # edge_indiceの内，被りがあるものを除去する
    processed_edges_indices = np.sort(processed_edges_indices, axis=1)
    ref_processed_edges_indices = processed_edges_indices.copy()
    unique_processed_edges_indices = np.unique(np.array(processed_edges_indices), axis=0)
    if unique_processed_edges_indices.shape[0] != processed_edges_indices.shape[0]:
        processed_edges_thickness = []
        for target_edge_indice in unique_processed_edges_indices:
            processed_edges_thickness.append(np.max(edges_thickness[find_edge_indice_index(target_edge_indice, ref_processed_edges_indices)]))
        processed_edges_thickness = np.array(processed_edges_thickness)
        processed_edges_indices = unique_processed_edges_indices
    else:
        processed_edges_thickness = edges_thickness

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


def conprocess_condition_edge_indices(processed_frozen_nodes, condition_frozen_nodes, processed_edges_indices, processed_edges_thickness):
    """固定ノード部分のうち，[2,3],[3,4]以外の[2,4]などを排除する
    """
    # processed_frozen_nodesとcondition_frozen_nodesのペアを作成する．
    # conprocess_seperate_edge_indice_procedureより，順序が紐づいていることを前提にしている．
    remove_indexes = []
    condition_frozen_nodes = np.array(condition_frozen_nodes)

    def cond(i, processed_frozen_nodes, condition_frozen_nodes):
        if ((i[0] in processed_frozen_nodes) & (i[1] in processed_frozen_nodes)):
            if (condition_frozen_nodes[processed_frozen_nodes == i[0]] != condition_frozen_nodes[processed_frozen_nodes == i[1]] - 1):
                return True
        return False
    remove_indexes = [cond(i, processed_frozen_nodes, condition_frozen_nodes) for i in processed_edges_indices]
    remove_indexes = np.where(remove_indexes)[0]
    processed_edges_indices = np.delete(processed_edges_indices, remove_indexes, 0)
    processed_edges_thickness = np.delete(processed_edges_thickness, remove_indexes, 0)

    return processed_edges_indices, processed_edges_thickness


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


def count_cross_points(nodes_pos, edges_indices):
    """グラフ内に交差しているエッジの交点の数がなんこあるかを出力する

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
    """
    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    edge_points_combinations = np.array([pair for pair in itertools.combinations(edge_points, 2)])
    pointsA = edge_points_combinations[:, 0, 0]
    pointsB = edge_points_combinations[:, 0, 1]
    pointsC = edge_points_combinations[:, 1, 0]
    pointsD = edge_points_combinations[:, 1, 1]
    bunbos = (pointsB[:, 0] - pointsA[:, 0]) * (pointsD[:, 1] - pointsC[:, 1]) - (pointsB[:, 1] - pointsA[:, 1]) * (pointsD[:, 0] - pointsC[:, 0])
    non_zero_index = bunbos != 0
    pointsA = pointsA[non_zero_index]
    pointsB = pointsB[non_zero_index]
    pointsC = pointsC[non_zero_index]
    pointsD = pointsD[non_zero_index]
    bunbos = bunbos[non_zero_index]
    vectorAC = np.stack([pointsC[:, 0] - pointsA[:, 0], pointsC[:, 1] - pointsA[:, 1]], axis=1)

    r = ((pointsD[:, 1] - pointsC[:, 1]) * vectorAC[:, 0] - (pointsD[:, 0] - pointsC[:, 0]) * vectorAC[:, 1]) / bunbos
    s = ((pointsB[:, 1] - pointsA[:, 1]) * vectorAC[:, 0] - (pointsB[:, 0] - pointsA[:, 0]) * vectorAC[:, 1]) / bunbos
    counts = np.count_nonzero(((0 < r) & (r < 1)) & ((0 < s) & (s < 1)))
    return counts


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


def calc_efficiency(input_nodes, input_vectors, output_nodes, output_vectors, displacement, E=1, b=0.2):
    denominator = np.sum([np.dot(input_vectors[i] / np.linalg.norm(input_vectors[i]), displacement[[input_node * 3 + 0, input_node * 3 + 1]]) for i, input_node in enumerate(input_nodes)])
    efficiency = np.dot(output_vectors / np.linalg.norm(output_vectors), displacement[[output_nodes[0] * 3 + 0, output_nodes[0] * 3 + 1]]) / denominator
    return efficiency


def calc_output_efficiency(input_nodes, input_vectors, output_nodes, output_vectors, displacement, E, A, L=1):
    # E: ヤング率
    # A: 代表面積
    # L: 代表長さ
    denominator = np.sum([np.linalg.norm(input_vectors[i]) for i, input_node in enumerate(input_nodes)])
    efficiency = np.dot(output_vectors / np.linalg.norm(output_vectors), displacement[[output_nodes[0] * 3 + 0, output_nodes[0] * 3 + 1]]) / denominator
    efficiency = efficiency * E * A / L
    return efficiency


def calc_misses_stress(stresses):
    # ミーゼス応力を求める
    tensile = stresses[:, [0, 3]]
    mage = stresses[:, [2, 5]]
    # 曲げ応力の最大の部分を取得
    rhox = np.abs(tensile + mage)
    rhox2 = np.abs(tensile - mage)
    index = rhox2 > rhox
    rhox[index] = rhox2[index]
    tauxy = stresses[:, [1, 4]]
    sqrt = np.sqrt((rhox / 2)**2 + tauxy**2)
    rho1 = rhox / 2 + sqrt
    rho2 = rhox / 2 - sqrt
    misses_stress = np.sqrt(1 / 2 * (rho1**2 + rho2**2 + (rho2 - rho1)**2))
    return misses_stress


def calc_axial_stress(stresses):
    # 軸方向の最大応力を求める
    tensile = stresses[:, [0, 3]]
    mage = stresses[:, [2, 5]]
    # 曲げ応力の最大の部分を取得
    rhox = np.abs(tensile + mage)
    rhox2 = np.abs(tensile - mage)
    index = rhox2 > rhox
    rhox[index] = rhox2[index]
    return rhox


def render_graph(nodes_pos, edges_indices, edges_thickness, save_path, display_number=False, edge_size=100):
    """グラフを図示
    Args:
        save_path (str, optional): 図を保存するパス.
        display_number (bool, optional): ノードに番号をつけるか付けないか. Defaults to False.
    """
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


def calc_length(node1_pos_x, node1_pos_y, node2_pos_x, node2_pos_y):
    return np.sqrt(np.power((node1_pos_x - node2_pos_x), 2) + np.power((node1_pos_y - node2_pos_y), 2))


def calc_volume(nodes_pos, edges_indices, edges_thickness):
    """グラフ構造が指定面積を占有している割合をエッジの太さ×長さを以ってして計算し，出力する

    Args:
        nodes_pos (np.array): (*,2)
        edges_indices (np.array): (*,2)
        edges_thickness (np.array): (*)
    """
    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    lengths = [calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points]
    return np.sum(lengths * edges_thickness)


def render_pixel_graph(nodes_pos, edges_indices, edges_thickness, save_path, pixel):
    edges = [[pixel * nodes_pos[edge_indice[0]], pixel * nodes_pos[edge_indice[1]],
              edge_thickness * pixel]
             for edge_indice, edge_thickness in zip(edges_indices, edges_thickness)]

    rho = make_bar_structure(pixel, pixel, edges)
    print(np.sum(rho))

    ny, nx = rho.shape
    x = np.arange(0, nx + 1)  # x軸の描画範囲の生成。
    y = np.arange(0, ny + 1)  # y軸の描画範囲の生成。
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    _ = plt.pcolormesh(X, Y, rho, cmap="binary")
    plt.axis("off")
    fig.savefig(save_path)
    plt.close()


def count_overlap_edge_pair(nodes_pos, edges_indices, edges_thickness, degree_threshhold=30):
    # それぞれのノードに対して，接続しているエッジの角度が隣同士で閾値以下のペアのものを数える関数
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes_pos)))
    edge_info = np.concatenate([edges_indices, edges_thickness.reshape((-1, 1))], axis=1)
    G.add_weighted_edges_from(edge_info)

    count = 0
    for target_node in G.nodes:
        target_edges_indices = np.array(list(G.edges(target_node)), dtype=int)  # 始点は与えたノード番号から始まる．例：26の場合，[26,3],[26,7]
        if target_edges_indices.shape[0] == 1:  # 一つしかエッジが存在しない場合
            continue
        else:
            edge_points_1 = nodes_pos[target_node]
            edge_points_2 = nodes_pos[target_edges_indices[:, 1]]
            vec = edge_points_2 - edge_points_1
            if target_edges_indices.shape[0] == 2:  # 2つしかエッジが存在しない場合
                rad = np.arctan2(vec[:, 0], vec[:, 1])
                rad_mod = np.mod(rad[0] - rad[1], 2 * np.pi)
                count += np.count_nonzero((rad_mod * 180 / np.pi < degree_threshhold) | ((360 - degree_threshhold) < rad_mod * 180 / np.pi))
            else:
                vec_rad_sort = np.sort(np.arctan2(vec[:, 0], vec[:, 1]))
                compare_vec_rad = np.stack([vec_rad_sort, np.roll(vec_rad_sort, 1)])
                rad_mod = np.mod(compare_vec_rad[0] - compare_vec_rad[1], 2 * np.pi)
                count += np.count_nonzero((rad_mod * 180 / np.pi < degree_threshhold) | ((360 - degree_threshhold) < rad_mod * 180 / np.pi))

    return count


def calc_minimum_perpendicular_line_length_edge_pair(nodes_pos, edges_indices, max_length=1.0):
    # それぞれのノードに対して，接続しているエッジの角度が隣同士でのものの垂線の長さの最小値を求める関数
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes_pos)))
    G.add_edges_from(edges_indices)

    perpendicular_line_lengths = []  # 垂線の足の長さを収納するリスト
    for target_node in G.nodes:
        target_edges_indices = np.array(list(G.edges(target_node)), dtype=int)  # 始点は与えたノード番号から始まる．例：26の場合，[26,3],[26,7]
        if target_edges_indices.shape[0] == 1:  # 一つしかエッジが存在しない場合
            continue
        else:
            edge_points_1 = nodes_pos[target_node]
            edge_points_2 = nodes_pos[target_edges_indices[:, 1]]
            vec = edge_points_2 - edge_points_1
            if target_edges_indices.shape[0] == 2:  # 2つしかエッジが存在しない場合
                rad = np.arctan2(vec[:, 0], vec[:, 1])
                rad_mod = np.mod(rad[0] - rad[1], 2 * np.pi)
                # 90度未満のエッジ同士の垂線の足の長さのみ求める
                if (rad_mod < np.pi / 2) | (3 / 2 * np.pi < rad_mod):
                    L1 = abs(np.linalg.norm(vec[0]) * np.sin(rad_mod))
                    L2 = abs(np.linalg.norm(vec[1]) * np.sin(rad_mod))
                    perpendicular_line_lengths.extend([L1, L2])

            else:
                vec_rad = np.arctan2(vec[:, 0], vec[:, 1])
                vec_arg_rad_sort = np.argsort(vec_rad)
                vec_rad_sort = vec_rad[vec_arg_rad_sort]
                vec_sort = vec[vec_arg_rad_sort]
                compare_vec_rad = np.stack([vec_rad_sort, np.roll(vec_rad_sort, 1)])
                compare_vec = np.stack([vec_sort, np.roll(vec_sort, 1, axis=0)])
                rad_mod = np.mod(compare_vec_rad[0] - compare_vec_rad[1], 2 * np.pi)
                # 90度未満のエッジ同士の垂線の足の長さのみ求める
                under_90_mask = (rad_mod < np.pi / 2) | (3 / 2 * np.pi < rad_mod)
                L1 = abs(np.linalg.norm(compare_vec[0, under_90_mask], axis=1) * np.sin(rad_mod[under_90_mask]))
                L2 = abs(np.linalg.norm(compare_vec[1, under_90_mask], axis=1) * np.sin(rad_mod[under_90_mask]))
                perpendicular_line_lengths.extend(L1.tolist())
                perpendicular_line_lengths.extend(L2.tolist())
    if len(perpendicular_line_lengths) != 0:
        return np.min(perpendicular_line_lengths)
    else:
        return max_length  # 最大の長さを与えることにする．


def calc_maximum_overlap_edge_length_ratio(nodes_pos, edges_indices, edges_thickness):
    # それぞれのノードに対して，重複している部分のエッジの長さに対する割合を求めていく．そして，その最大値を出力する．
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes_pos)))
    edge_info = np.concatenate([edges_indices, edges_thickness.reshape((-1, 1))], axis=1)
    G.add_weighted_edges_from(edge_info)

    perpendicular_line_lengths = []  # 垂線の足の長さを収納するリスト
    for target_node in G.nodes:
        target_edges_indices = np.array(list(G.edges(target_node)), dtype=int)  # 始点は与えたノード番号から始まる．例：26の場合，[26,3],[26,7]
        if target_edges_indices.shape[0] == 1:  # 一つしかエッジが存在しない場合
            continue
        else:
            edge_points_1 = nodes_pos[target_node]
            edge_points_2 = nodes_pos[target_edges_indices[:, 1]]
            vec = edge_points_2 - edge_points_1
            if target_edges_indices.shape[0] == 2:  # 2つしかエッジが存在しない場合
                rad = np.arctan2(vec[:, 0], vec[:, 1])
                rad_mod = np.mod(rad[0] - rad[1], 2 * np.pi)
                if (0 == rad_mod) | (rad_mod == 2 * np.pi):  # もし0度の完全なる重複が存在した場合，1000を返す
                    return 1000
                # 90度未満のエッジ同士の垂線の足の長さのみ求める
                if ((0 < rad_mod) & (rad_mod < np.pi / 2)) | ((3 / 2 * np.pi < rad_mod) & (rad_mod < 2 * np.pi)):
                    L1 = abs(np.linalg.norm(vec[0]))
                    L2 = abs(np.linalg.norm(vec[1]))
                    target_edges_indices = np.sort(target_edges_indices)
                    width1 = G.edges[target_edges_indices[0]]["weight"]
                    width2 = G.edges[target_edges_indices[1]]["weight"]
                    sin_theta = np.abs(np.sin(rad_mod))  # rad_modを強制的に0<rad_mod < np.pi / 2の条件にしている．
                    cos_theta = np.cos(rad_mod)
                    A1 = np.arcsin((width1 * sin_theta) / (np.sqrt(width1**2 + width2**2 + 2 * width1 * width2 * cos_theta)))
                    d1 = width1 / (2 * np.tan(A1))
                    A2 = np.arcsin((width2 * sin_theta) / (np.sqrt(width2**2 + width1**2 + 2 * width1 * width2 * cos_theta)))
                    d2 = width2 / (2 * np.tan(A2))
                    perpendicular_line_lengths.extend([d1 / L1, d2 / L2])

            else:
                target_edges_indices = np.sort(target_edges_indices)

                widths = np.array([G.edges[i]["weight"] for i in target_edges_indices])
                vec_rad = np.arctan2(vec[:, 1], vec[:, 0])  # 各エッジの向いている角度.y,xの順で指定することに注意
                vec_arg_rad_sort = np.argsort(vec_rad)  # 小さい順にソート
                vec_rad_sort = vec_rad[vec_arg_rad_sort]
                vec_sort = vec[vec_arg_rad_sort]
                widths_sort = widths[vec_arg_rad_sort]  # 回転の順番通り

                compare_vec_rad = np.stack([vec_rad_sort, np.roll(vec_rad_sort, 1)])  # ソートされたエッジと，その一つ時計回りのエッジの角度を収納する
                compare_vec = np.stack([vec_sort, np.roll(vec_sort, 1, axis=0)])
                compare_widths = np.stack([widths_sort, np.roll(widths_sort, 1)])
                rad_mod = np.mod(compare_vec_rad[0] - compare_vec_rad[1], 2 * np.pi)
                if np.any((0 == rad_mod) | (rad_mod == 2 * np.pi)):  # もし0度の完全なる重複が存在した場合，1000を返す
                    return 1000
                # 90度未満のエッジ同士の垂線の足の長さのみ求める
                under_90_mask = ((0 < rad_mod) & (rad_mod < np.pi / 2)) | ((3 / 2 * np.pi < rad_mod) & (rad_mod < 2 * np.pi))
                sin_theta = np.abs(np.sin(rad_mod[under_90_mask]))
                cos_theta = np.cos(rad_mod[under_90_mask])
                L1 = abs(np.linalg.norm(compare_vec[0, under_90_mask], axis=1))  # 反時計回り側のエッジの長さ
                L2 = abs(np.linalg.norm(compare_vec[1, under_90_mask], axis=1))  # 時計回り側のエッジの長さ
                width1 = compare_widths[0, under_90_mask]
                width2 = compare_widths[1, under_90_mask]
                A1 = np.arcsin((width1 * sin_theta) / (np.sqrt(width1**2 + width2**2 + 2 * width1 * width2 * cos_theta)))
                d1 = width1 / (2 * np.tan(A1))  # 反時計回り側のエッジの重複部分の長さ
                A2 = np.arcsin((width2 * sin_theta) / (np.sqrt(width2**2 + width1**2 + 2 * width1 * width2 * cos_theta)))
                d2 = width2 / (2 * np.tan(A2))  # 時計回り側のエッジの重複部分の長さ
                perpendicular_line_lengths.extend((d1 / L1).tolist())
                perpendicular_line_lengths.extend((d2 / L2).tolist())
    if len(perpendicular_line_lengths) != 0:
        return np.max(perpendicular_line_lengths)
    else:
        return 0.0  # 一つも重複していない場合，0.0を出力する


def calc_segment_line_dist(nodes_pos, edges_indices):
    # 各エッジ（線分）と各ノード（点）との距離を(edges,nodes)形式で出力する．
    each_node_edges_indices_length = np.zeros((edges_indices.shape[0], nodes_pos.shape[0]))
    for i, edge_indices in enumerate(edges_indices):
        edge_points = np.stack([nodes_pos[edge_indices[0]], nodes_pos[edge_indices[1]]])
        vec1 = edge_points[0] - edge_points[1]  # エッジを構成する節点同士のベクトル
        vec2 = nodes_pos - edge_points[1]  # 各ノードとエッジを構成する節点のベクトル
        vec_dot = vec1[0] * vec2[:, 0] + vec1[1] * vec2[:, 1]  # 内積
        # 線分上に近接点が存在するエッジを抽出
        segment_exist_mask = (0 <= vec_dot) & (vec_dot <= np.linalg.norm(vec1)**2)
        segment_distances = np.abs(vec1[0] * vec2[segment_exist_mask, 1] - vec1[1] * vec2[segment_exist_mask, 0]) / (np.linalg.norm(vec1))
        # 線分上に近接点が存在しないエッジは，エッジの端点と点との距離を抽出
        vec3 = nodes_pos - edge_points[0]
        node_segment_node_distances = np.min(np.stack([np.linalg.norm(vec2[~segment_exist_mask], axis=1), np.linalg.norm(vec3[~segment_exist_mask], axis=1)]), axis=0)
        each_node_edges_indices_length[i, segment_exist_mask] = segment_distances
        each_node_edges_indices_length[i, ~segment_exist_mask] = node_segment_node_distances
    return each_node_edges_indices_length


def calc_minimum_segment_line_dist_ratio(nodes_pos, edges_indices, edges_thickness):
    # 各エッジ（線分）と各ノード（点）との距離に対するエッジの太さの割合の最小値を求める
    each_node_edges_indices_length = calc_segment_line_dist(nodes_pos, edges_indices)
    each_node_max_thickness = []  # 各ノードに関わるエッジの太さの内，最大のものを収納するリスト
    for i in range(nodes_pos.shape[0]):
        i_include_edges_indices_index = np.argwhere((edges_indices[:, 0] == i) | (edges_indices[:, 1] == i)).squeeze()
        each_node_max_thickness.append(np.max(edges_thickness[i_include_edges_indices_index]))
    average_edges_thickness = np.tile(edges_thickness.reshape((-1, 1)), nodes_pos.shape[0])
    average_edges_thickness = (average_edges_thickness + np.array(each_node_max_thickness)) / 2
    edge_thick_ratio = each_node_edges_indices_length / average_edges_thickness
    mask = np.ones(edge_thick_ratio.shape, dtype=bool)  # エッジを構成するノードを抜いたmask
    for i, edge_indices in enumerate(edges_indices):
        mask[i, edge_indices] = False
    return np.min(edge_thick_ratio[mask])


def calc_equation(point1, point2):
    # point1とpoint2を通る方程式の係数を求める．
    # 返す係数はy=ax+bのうち，(a,b)の順に返す．
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = (point2[0] * point1[1] - point1[0] * point2[1]) / (point2[0] - point1[0])
    return (a, b)


def calc_lengths(nodes_pos, edges_indices):
    edge_points = np.array([np.stack([nodes_pos[edges_indice[0]], nodes_pos[edges_indice[1]]]) for edges_indice in edges_indices])
    lengths = calc_length(edge_points[:, 0, 0], edge_points[:, 0, 1], edge_points[:, 1, 0], edge_points[:, 1, 1])
    return lengths


def calc_maximum_buckling_force_ratio(edges_thickness, nodes_pos, edges_indices, E, b, stresses, input_nodes):
    lengths = calc_lengths(nodes_pos, edges_indices)
    A = edges_thickness * b
    I = (A * edges_thickness**2) / 12
    n = 4  # 端末係数．今回の場合，両端回転・変位固定
    P = n * (np.pi**2) * E * I / (lengths**2)  # 座屈荷重
    tensile = stresses[:, [0, 3]][:, 1]  # 軸荷重,負だったら圧縮
    force_x = tensile / A
    compress_mask = tensile < 0
    katahoukotei_mask = (edges_indices[:, 0] == input_nodes[0]) | (edges_indices[:, 1] == input_nodes[0])
    P[katahoukotei_mask] = P[katahoukotei_mask] / 16
    return np.min(-P[compress_mask] / force_x[compress_mask])
