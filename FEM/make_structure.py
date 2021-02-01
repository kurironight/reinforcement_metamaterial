import numpy as np


def make_bar_structure(x_size, y_size, edges):
    """edgesを基に，バーを用いた構造を作成する

    Args:
        x_size (float): 目標とする1/4スケールの構造物のx方向のブロック数
        y_size (float): 目標とする1/4スケールの構造物のy方向のブロック数
        edges (list): [エッジの始点，終点，太さ]のリスト

    Returns:
        np.array (y_size * x_size): バー構造
    """
    rho = np.zeros([y_size, x_size], dtype=np.float64)
    for edge in edges:
        put_bar(rho, edge[0], edge[1], edge[2])
    return rho


def put_bar(rho, start_point, end_point, width):
    assert start_point[0] >= 0 and start_point[0] <= rho.shape[1], 'start_point x index {} must be 0~{}'.format(
        start_point[0], rho.shape[1])
    assert start_point[1] >= 0 and start_point[1] <= rho.shape[0], 'start_point y index {} must be 0~{}'.format(
        start_point[1], rho.shape[0])
    assert end_point[0] >= 0 and end_point[0] <= rho.shape[1], 'end_point x index {} must be 0~{}'.format(
        end_point[0], rho.shape[1])
    assert end_point[1] >= 0 and end_point[1] <= rho.shape[0], 'end_point y index {} must be 0~{}'.format(
        end_point[1], rho.shape[0])
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    # 端点が，始点終点が一緒の場合は何もしない
    if np.all(start_point == end_point):
        return rho
    else:
        edge_point1 = end_point+(end_point-start_point) / \
            np.linalg.norm(end_point - start_point, ord=2)*0.5
        edge_point2 = start_point + \
            (start_point-end_point) / \
            np.linalg.norm(end_point - start_point, ord=2)*0.5

    # xの方でメッシュの座標は[0,.....,rho.shape[1]]．なお，数はrho.shape[1]＋１
    x_index = np.linspace(0, rho.shape[1], num=rho.shape[1], dtype=np.float64)
    y_index = np.linspace(0, rho.shape[0], num=rho.shape[0], dtype=np.float64)
    xx, yy = np.meshgrid(x_index, y_index)
    if (end_point[0]-start_point[0]) != 0:  # x=8等のような直線の式にならない場合
        m = (end_point[1]-start_point[1])/(end_point[0]-start_point[0])
        n = end_point[1]-m*end_point[0]
        d = np.abs(yy-m*xx-n)/np.sqrt(1+np.power(m, 2))
        # 垂線の足を求める
        X = (m*(yy-n)+xx)/(np.power(m, 2)+1)
        Y = m*X+n
        # バーを配置できる条件を満たすインデックスを求める
        X_on_segment = np.logical_and(min(
            edge_point1[0], edge_point2[0]) <= X, X <= max(edge_point1[0], edge_point2[0]))
        Y_on_segment = np.logical_and(min(
            edge_point1[1], edge_point2[1]) <= Y, Y <= max(edge_point1[1], edge_point2[1]))
        on_segment = np.logical_and(X_on_segment, Y_on_segment)
        in_distance = d <= width/2
        meet_index = np.logical_and(on_segment, in_distance)
    else:  # x=8等の場合
        d = np.abs(end_point[0]-xx)
        # 垂線の足を求める
        X = end_point[0]
        Y = yy
        # バーを配置できる条件を満たすインデックスを求める
        Y_on_segment = np.logical_and(min(
            edge_point1[1], edge_point2[1]) <= Y, Y <= max(edge_point1[1], edge_point2[1]))
        in_distance = d <= width/2
        meet_index = np.logical_and(Y_on_segment, in_distance)
    rho[meet_index] = 1
    return rho
