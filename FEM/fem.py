import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

E = 1  # ヤング率
# rho = np.ones((2, 2))

rho = np.array([[1, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1], ]
               )
po = 0.3  # ポアソン比
pnl = 3  # ペナルティパラメータ
t = 1  # 要素の厚み方向の長さ
cut_thresh = 10**(-2)  # 材料が存在するかどうかの基準密度


def calc_E(rho, cut_thresh=cut_thresh):
    """縦弾性係数を求める

    Args:
        rho (np.array): 材料密度分布
        cut_thresh (float, optional): 材料が存在するかしないかを決めるもの. Defaults to cut_thresh.

    Returns:
        numpy.float: 縦弾性係数
    """
    rho[rho < cut_thresh] = 0
    ny, nx = rho.shape
    # 境界条件　左端x軸方向固定　上端y軸方向固定 右端　x正方向に分散荷重
    FixDOF_left_edge = list(range(1, 2 * (ny + 1), 2))
    FixDOF_up_edge = list(range(2, 2 * (nx + 1) * (ny + 1), 2 * (ny + 1)))
    FixDOF = FixDOF_left_edge+FixDOF_up_edge

    F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
    F_index = np.where(rho[:, -1] != 0)[0]*2 + 2 * \
        nx * (ny + 1)  # python用のindexにした
    F[F_index] += 1/2  # 要素端部においての力が1/2になるようにする為
    F[F_index+2] += 1/2
    F = F/np.sum(F)  # 分散荷重の大きさを正規化

    # 有限要素法適用
    U = FEM(rho, FixDOF, F)
    U = U.reshape([-1, 2])
    element_exist_index = np.where(
        rho[:, -1] != 0)[0] + nx * (ny + 1)  # 密度が１の要素における変位のみにフォーカス
    element_exist_index = np.unique(np.concatenate(
        [element_exist_index, element_exist_index+1]))
    change = np.mean(U[element_exist_index, 0])
    E = (1/ny)/(change/nx)  # 応力/ひずみ
    return E


def calc_G(rho, cut_thresh=cut_thresh):
    """せん断弾性係数を求める

    Args:
        rho (np.array): 材料密度分布（1/4の構造を示している）
        cut_thresh (float, optional): 材料が存在するかしないかを決めるもの. Defaults to cut_thresh.

    Returns:
        numpy.float: せん断弾性係数
    """
    rho[rho < cut_thresh] = 0
    left_rho = np.fliplr(rho).copy()
    down_rho = np.concatenate([left_rho, rho], 1)
    up_rho = np.flipud(down_rho).copy()
    whole_rho = np.concatenate([up_rho, down_rho], 0)
    ny, nx = whole_rho.shape
    # 境界条件　左端固定　右端　y正方向にせん断荷重
    FixDOF_left_edge = list(range(1, 2 * (ny + 1)))
    FixDOF = FixDOF_left_edge

    F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
    F_index = np.where(whole_rho[:, -1] != 0)[0] * \
        2 + 2 * nx * (ny + 1)  # python用のindexにした
    F[F_index+1] += 1/2  # 要素端部においての力が1/2になるようにする為
    F[F_index+3] += 1/2
    F = F/np.sum(F)  # 分散荷重の大きさを正規化

    # 有限要素法適用
    U = FEM(whole_rho, FixDOF, F)
    U = U.reshape([-1, 2])
    element_exist_index = np.where(
        whole_rho[:, -1] != 0)[0] + nx * (ny + 1)  # 密度が１の要素における変位のみにフォーカス
    element_exist_index = np.unique(np.concatenate(
        [element_exist_index, element_exist_index+1]))
    change = np.mean(U[element_exist_index, 1])
    G = (1/nx)/(change/nx)  # せん断応力/せん断ひずみ
    return G


def FEM(rho, FixDOF, F):
    """材料密度分布がRHOの時に，固定条件FIXDOF及び荷重条件Fに伴ってFEMを行い，変位を求める．
    節点番号は1~2 * (nx + 1) * (ny + 1)まで．
    １，３，５など奇数はx方向を示す．


    Args:
        rho (np.array): 二次元の材料密度分布(ny*nx)
        FixDOF (np.array): 2*(ny+1)*(nx+1):左下から上，右順に節点を表す．
        F (np.array): 2*(ny+1)*(nx+1)：正の値はｙ軸方向，ｘ軸方向の力を示す．

    Returns:
        [np.array]: 2*(ny+1)*(nx+1)
    """
    K = make_K_mat(rho)
    ny, nx = rho.shape
    U = np.zeros((2 * (nx + 1) * (ny + 1)), dtype=np.float64)
    # Boundary Condition
    #F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
    # F[9-1]=10

    # 節点番号は1~2 * (nx + 1) * (ny + 1)まで
    # FixDOF = list(range(1, 2 * (ny + 1)+1))
    FreeDOF = list(range(1, 2 * (nx + 1) * (ny + 1)+1))
    for i in FixDOF:
        FreeDOF.remove(i)
    # indexを示す為
    FreeDOF = np.array(FreeDOF) - 1
    FixDOF = np.array(FixDOF) - 1
    target_K = csr_matrix(K[np.ix_(FreeDOF, FreeDOF)])
    U[FreeDOF] = spsolve(
        target_K,   F[FreeDOF], use_umfpack=True)
    U[FixDOF] = 0
    return U


def FEM_displacement(rho, FixDOF, F, disp):
    """材料密度分布がRHOの時に，固定条件FIXDOF及び荷重条件Fに伴ってFEMを行い，変位を求める．
    節点番号は1~2 * (nx + 1) * (ny + 1)まで．
    １，３，５など奇数はx方向を示す．


    Args:
        rho (np.array): 二次元の材料密度分布(ny*nx)
        FixDOF (np.array): 2*(ny+1)*(nx+1):左下から上，右順に節点を表す．
        F (np.array): 2*(ny+1)*(nx+1)：正の値はｙ軸方向，ｘ軸方向の力を示す．
        disp (np.array): 2*(ny+1)*(nx+1)：正の値はｙ軸方向，ｘ軸方向の変位を示す．

    Returns:
        [np.array]: 2*(ny+1)*(nx+1)
    """
    K = make_K_mat(rho)
    ny, nx = rho.shape
    U = disp.copy()

    disp_indices = np.where(disp != 0)
    for disp_indice in disp_indices:
        KU = K[:, disp_indice]*U[disp_indice]
        F = F-KU.flatten()  # F-KUしている
        F[disp_indice] = U[disp_indice]
        K[:, disp_indice] = 0
        K[disp_indice, :] = 0
        K[disp_indice, disp_indice] = 1

    FreeDOF = list(range(1, 2 * (nx + 1) * (ny + 1)+1))
    for i in FixDOF:
        FreeDOF.remove(i)
    # indexを示す為
    FreeDOF = np.array(FreeDOF) - 1
    FixDOF = np.array(FixDOF) - 1
    target_K = csr_matrix(K[np.ix_(FreeDOF, FreeDOF)])
    U[FreeDOF] = spsolve(
        target_K,   F[FreeDOF], use_umfpack=True)
    U[FixDOF] = 0
    return U


def make_K_mat(rho):
    # 全体のK行列を作成
    ny, nx = rho.shape
    K = np.zeros((2 * (nx + 1) * (ny + 1), 2 * (nx + 1)
                  * (ny + 1)), dtype=np.float64)
    for y in range(1, ny+1):
        for x in range(1, nx+1):
            n1 = (ny + 1) * (x - 1) + y
            n2 = (ny + 1) * x + y
            # 要素を構成する要素番号(x,yそれぞれを含む，奇数がx，偶数がy)
            elem = np.array([2*n1-1, 2*n1, 2*n2-1, 2*n2,
                             2*n2+1, 2*n2+2, 2*n1+1, 2*n1+2])
            xc = [x - 1, x, x, x - 1]  # x節点の範囲は0~nx
            yc = [y - 1, y - 1, y, y]  # y節点の範囲は0~ny
            K_mat = Kmat_pl4(xc, yc, po, rho[y-1, x-1])
            elem -= 1  # indexを指定する為
            K[np.ix_(elem, elem)] += K_mat  # 深いコピーになる
    return K


def dmat_pl4(Eelem, po, nstr=1):
    # Dマトリックス作成
    # nstr=0の時，平面歪み，1の時，平面応力
    D_mat = np.zeros((3, 3), dtype=np.float64)
    if nstr == 0:  # plane strain
        D_mat[0, 0] = 1-po
        D_mat[0, 1] = po
        D_mat[1, 0] = po
        D_mat[1, 1] = 1-po
        D_mat[2, 2] = 0.5*(1-2*po)
        D_mat = Eelem/(1+po)/(1-2*po)*D_mat
    else:  # plane stress
        D_mat[0, 0] = 1
        D_mat[0, 1] = po
        D_mat[1, 0] = po
        D_mat[1, 1] = 1
        D_mat[2, 2] = 0.5*(1-po)
        D_mat = Eelem/(1-po**2)*D_mat
    return D_mat


def Kmat_pl4(xc, yc, po, rho, t=t):
    """各四角形事のKマトリックスを作成

    Args:
        xc ([list]): 要素を構成するx座標のリスト
        yc ([list]): 要素を構成するy座標のリスト
        po (float): ポアソン比
        rho(np.float): 要素の密度
        t (int, optional): z方向の要素の厚み.
    """
    Emin = 10**(-5)
    Eelem = (E-Emin)*rho**pnl+Emin  # 密度に対しての材料物性値の調整
    K_mat = np.zeros((8, 8))
    D_mat = dmat_pl4(Eelem, po)
    point, weight = gauss_point()
    for i in range(4):
        B_mat, detJ = bmat_pl4(point[i, 0], point[i, 1], xc, yc)
        K_mat = K_mat + weight[i] * \
            np.dot(B_mat.T, np.dot(D_mat, B_mat)) * t * detJ

    return K_mat


def gauss_point():
    point = np.array([[-3 ** (-1 / 2), -3 ** (-1 / 2)],
                      [3 ** (-1 / 2), -3 ** (-1 / 2)],
                      [3 ** (-1 / 2), 3 ** (-1 / 2)],
                      [-3 ** (-1 / 2), 3 ** (-1 / 2)]], dtype=np.float64)
    weight = [1, 1, 1, 1]
    return point, weight


def bmat_pl4(a, b, xc, yc):
    """Bマトリックスとヤコビアンを求める

    Args:
        a (number): ガウスポイントの座標
        b (number): ガウスポイントの座標
        xc (list): 要素を構成する節点のx座標
        yc (list): 要素を構成する節点のy座標
    """
    # Bマトリックスとヤコビアンを導出
    # aとbには,ガウスpointが入る
    x1, x2, x3, x4 = xc
    y1, y2, y3, y4 = yc
    bm = np.zeros((3, 8))
    # dN/da,dN/db
    dn1a = -0.25*(1.0-b)
    dn2a = 0.25*(1.0-b)
    dn3a = 0.25*(1.0+b)
    dn4a = -0.25*(1.0+b)
    dn1b = -0.25*(1.0-a)
    dn2b = -0.25*(1.0+a)
    dn3b = 0.25*(1.0+a)
    dn4b = 0.25*(1.0-a)
    # Jacobi matrix and det(J)
    J11 = dn1a*x1+dn2a*x2+dn3a*x3+dn4a*x4
    J12 = dn1a*y1+dn2a*y2+dn3a*y3+dn4a*y4
    J21 = dn1b*x1+dn2b*x2+dn3b*x3+dn4b*x4
    J22 = dn1b*y1+dn2b*y2+dn3b*y3+dn4b*y4
    detJ = J11*J22-J12*J21
    # [B]=[dN/dx][dN/dy]
    bm[0, 0] = J22*dn1a-J12*dn1b
    bm[0, 2] = J22*dn2a-J12*dn2b
    bm[0, 4] = J22*dn3a-J12*dn3b
    bm[0, 6] = J22*dn4a-J12*dn4b
    bm[1, 1] = -J21*dn1a+J11*dn1b
    bm[1, 3] = -J21*dn2a+J11*dn2b
    bm[1, 5] = -J21*dn3a+J11*dn3b
    bm[1, 7] = -J21*dn4a+J11*dn4b
    bm[2, 0] = -J21*dn1a+J11*dn1b
    bm[2, 1] = J22*dn1a-J12*dn1b
    bm[2, 2] = -J21*dn2a+J11*dn2b
    bm[2, 3] = J22*dn2a-J12*dn2b
    bm[2, 4] = -J21*dn3a+J11*dn3b
    bm[2, 5] = J22*dn3a-J12*dn3b
    bm[2, 6] = -J21*dn4a+J11*dn4b
    bm[2, 7] = J22*dn4a-J12*dn4b
    bm = bm/detJ
    return bm, detJ
