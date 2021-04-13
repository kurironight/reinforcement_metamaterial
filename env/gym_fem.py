import numpy as np
from FEM.make_structure import make_bar_structure
from FEM.fem import FEM, FEM_displacement
import matplotlib.pyplot as plt
from .gym_metamech import MetamechGym, MAX_EDGE_THICKNESS
import cv2
import os

PIXEL = 50


class FEMGym(MetamechGym):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, edges_indices, edges_thickness):
        super(FEMGym, self).__init__(node_pos, 0, 0,
                                     0, 0, 0, edges_indices, edges_thickness)

        self.pixel = PIXEL
        self.max_edge_thickness = MAX_EDGE_THICKNESS

        # condition for calculation
        ny = self.pixel
        nx = self.pixel
        Y_DOF = np.linspace(2, 2 * (nx) * (ny + 1) + 2, num=nx + 1, dtype=np.int32)
        X_DOF = np.linspace(1, 2 * (nx) * (ny + 1) + 1, num=nx + 1, dtype=np.int32)
        self.FIXDOF = np.concatenate([X_DOF, Y_DOF])
        self.displace_DOF = 2 * (nx + 1) * (ny + 1)  # 強制変位を起こす場所
        """
        力を付加する場合の条件
        F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
        F[self.displace_DOF-1] = -1
        self.F = F
        """

        displacement_condition = np.zeros(
            (2 * (nx + 1) * (ny + 1)), dtype=np.float64)
        displacement_condition[self.displace_DOF - 1] = -1
        self.displacement_condition = displacement_condition

        # 構造が繋がっているかを確認する時，確認するメッシュ位置のindex
        self.check_output_mesh_index = (ny - 1, 0)
        self.check_input_mesh_index = (ny - 1, nx - 1)
        self.check_freeze_mesh_index = (0, int(nx / 2))

        # efficiencyを計算するとき，節点変位を確認する出力部の節点のDOF
        self.check_x_output_node_DOF = 2 * (ny + 1) - 1
        self.check_y_output_node_DOF = 2 * (ny + 1)
        # efficiencyを計算するとき，出力部の目標ベクトル
        self.output_vector = np.array([-1, 0])

    def extract_rho_for_fem(self):
        nodes_pos, edges_indices, edges_thickness, _ = self.extract_node_edge_info()

        edges = [[self.pixel * nodes_pos[edges_indice[0]], self.pixel * nodes_pos[edges_indice[1]],
                  self.max_edge_thickness * edge_thickness]
                 for edges_indice, edge_thickness in zip(edges_indices, edges_thickness)]

        rho = make_bar_structure(self.pixel, self.pixel, edges)

        return rho

    def calculate_simulation(self):
        rho = self.extract_rho_for_fem()
        """
        力を付加する場合の計算方法
        U = FEM(rho, self.FIXDOF, self.F)
        displacement = np.array(
            [U[self.check_x_output_node_DOF-1], U[self.check_y_output_node_DOF-1]])
        efficiency = np.dot(self.output_vector, displacement)
        
        print("力：\n", efficiency)
        """
        U = FEM_displacement(rho, self.FIXDOF, np.zeros(
            (2 * (self.pixel + 1) * (self.pixel + 1)), dtype=np.float64), self.displacement_condition)

        # actuator.pyより引用
        displacement = np.array(
            [U[self.check_x_output_node_DOF - 1], U[self.check_y_output_node_DOF - 1]])
        efficiency = np.dot(self.output_vector, displacement)

        return efficiency

    def confirm_graph_is_connected(self):
        # グラフが全て接続しているか確認

        rho = self.extract_rho_for_fem().astype(np.uint8)
        _, markers = cv2.connectedComponents(rho)
        if markers[self.check_output_mesh_index] == markers[self.check_input_mesh_index] & \
                markers[self.check_output_mesh_index] == markers[self.check_freeze_mesh_index]:
            return True
        else:
            return False

    # 環境の描画
    def render(self, save_path="image/image.png"):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        rho = self.extract_rho_for_fem()
        ny, nx = rho.shape
        x = np.arange(0, nx + 1)  # x軸の描画範囲の生成。
        y = np.arange(0, ny + 1)  # y軸の描画範囲の生成。
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        _ = plt.pcolormesh(X, Y, rho, cmap="binary")
        plt.axis("off")
        fig.savefig(save_path)
        plt.close()
