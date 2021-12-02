import numpy as np
import sympy as sym
from scipy.misc import derivative
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calc_function_reduction_rate(a=8e-6, b=- 0.0054, c=1.3114, target_length=6.5):
    # target_lengthはハエトリグサの中心部の想定している長さ
    x = sym.symbols("x")
    original_central_curve_function = a * x**(3) + b * x**(2) + c * x
    curve_length_func = sym.sqrt((1 + (sym.diff(original_central_curve_function) * sym.diff(original_central_curve_function))))  # 曲線の長さを求めるための式
    res = sym.integrate(curve_length_func, (x, 0, 207))  # このままだとまだ積分式のまま
    length = res.evalf()
    reduction_rate = length / target_length
    return reduction_rate


def make_node(vertical_division, horizontal_division, vertical_length=6.5, horizontal_length=15,
              curvature=0.04, left_thick=1, right_thick=0.5, plot=False):
    """ハエトリグサの構造を8点ソリッドシェル要素として表現する関数.
    これは仮定として全ての部分における曲率は一定

    Args:
        vertical_division (int)): 中肋方向にいくつメッシュを分割するか
        horizontal_division (int)): 中肋に垂直方向にいくつメッシュを分割するか
        vertical_length (float, optional): 中肋方向の長さ. Defaults to 6.5.
        horizontal_length (int, optional): 中肋に垂直方向の長さ. Defaults to 15.
        curvature (float, optional): 中肋に垂直方向の曲率. Defaults to 0.04.
        left_thick (int, optional): 葉の根本の中肋の太さ. Defaults to 1.
        right_thick (float, optional): 葉の先の中肋の太さ. Defaults to 0.5.
        plot (bool, optional): 三次元点をプロットするか否か. Defaults to False.

    Returns:
        [type]: 低い方の節点と高い方の節点の位置座標．
        並び方としては，それぞれ根本の-z方向からz方向に1,2,3..horizontal_division．次に，根本から葉の先方向にhorizontal_division+1,..,2*horizontal_division.
    """
    #
    reduction_rate = 37.4280922360231  # calc_function_reduction_rateの結果を利用
    max_x = 207 / reduction_rate  # xの最大座標

    R = 1 / curvature
    max_deg = horizontal_length / (2 * R)  # 奥方向の最大角度
    x_coord_space = np.linspace(0, max_x, horizontal_division).reshape((-1, 1))  # x座標のリスト
    deg_space = np.linspace(-max_deg, max_deg, horizontal_division).reshape((-1, 1))  # 角度のリスト

    lower_points = []
    higher_points = []

    def central_curve_function(x):
        return (8e-6 * (x * reduction_rate)**3 - 0.0054 * (x * reduction_rate)**2 + 1.3114 * (x * reduction_rate)) / reduction_rate  # 縮小倍率を適用した関数

    for x_loc in x_coord_space:
        x_loc = x_loc.squeeze()
        center_loc = np.array([x_loc, central_curve_function(x_loc), 0])
        diff_x = derivative(central_curve_function, x_loc, dx=1e-6)
        vec_a = np.array([1, diff_x, 0])
        unit_vec_a = vec_a / np.linalg.norm(vec_a)
        unit_vec_b = np.array([0, 0, 1])
        unit_vec_c = np.cross(unit_vec_a, unit_vec_b)
        center_of_circle_loc = center_loc + R * unit_vec_c

        # 奥方向の点の座標を-zから+zにかけて求めていく．
        lower_point = center_of_circle_loc + R * np.cos(deg_space) * -unit_vec_c + R * np.sin(deg_space) * unit_vec_b
        higher_point = center_of_circle_loc + (R + 1.5) * np.cos(deg_space) * -unit_vec_c + (R + 1.5) * np.sin(deg_space) * unit_vec_b
        lower_points.append(lower_point)
        higher_points.append(higher_point)

    lower_points = np.array(lower_points).reshape((-1, 3))
    higher_points = np.array(higher_points).reshape((-1, 3))

    if plot:
        # 描画エリアの作成
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # numpyを使ってXYZの値を設定
        x = lower_points[:, 0]
        y = lower_points[:, 1]
        z = lower_points[:, 2]

        hx = higher_points[:, 0]
        hy = higher_points[:, 1]
        hz = higher_points[:, 2]

        # 散布図の作成
        ax.scatter(x, y, z, s=40, c="red")
        ax.scatter(hx, hy, hz, s=40, c="blue")

        # 軸ラベルのサイズと色を設定
        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("y", size=15, color="black")
        ax.set_zlabel("z", size=15, color="black")

        # 描画
        plt.show()

    return lower_points, higher_points
