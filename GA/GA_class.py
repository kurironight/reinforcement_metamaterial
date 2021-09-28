from platypus import NSGAII, Problem, nondominated, Integer, Real, Binary, \
    CompoundOperator, SBX, HUX, PM, BitFlip
from .condition import condition, condition_only_input_output
from tools.lattice_preprocess import make_main_node_edge_info
from tools.graph import preprocess_graph_info, separate_same_line_procedure, \
    conprocess_seperate_edge_indice_procedure, seperate_cross_line_procedure, calc_efficiency,\
    remove_node_which_nontouchable_in_edge_indices, render_graph, check_cross_graph, count_cross_points
import numpy as np
from .utils import make_edge_thick_triu_matrix, make_adj_triu_matrix
import networkx as nx
import os
from FEM.bar_fem import barfem, barfem_mapdl
from tools.graph import calc_length, calc_volume
from tools.save import save_graph_info_npy


class Barfem_GA(Problem):
    def __init__(self, free_node_num, fix_node_num, max_edge_thickness=0.05, min_edge_thickness=0.01, condition_edge_thickness=0.05):
        self.condition_edge_thickness = condition_edge_thickness
        self.condition_nodes_pos, self.input_nodes, self.input_vectors, self.output_nodes, \
            self.output_vectors, self.frozen_nodes, self.condition_edges_indices, self.condition_edges_thickness\
            = make_main_node_edge_info(*condition(), condition_edge_thickness=self.condition_edge_thickness)  # スレンダー比を考慮し，長さ方向に対して1/20の値の幅にした
        self.fix_node_num = fix_node_num
        self.free_node_num = free_node_num
        self.input_output_node_num = 2  # 入力ノードと出力ノードの合計数
        self.node_num = free_node_num + fix_node_num + self.input_output_node_num  # 条件ノード込のノードの数
        condition_node_num = self.condition_nodes_pos.shape[0]
        self.gene_node_pos_num = (self.node_num - condition_node_num) * 2
        self.gene_edge_thickness_num = int(self.node_num * (self.node_num - 1) / 2)
        self.gene_edge_indices_num = self.gene_edge_thickness_num
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1)
        self.max_edge_thickness = max_edge_thickness
        self.min_edge_thickness = min_edge_thickness
        assert min_edge_thickness < self.max_edge_thickness, "max_edge_thickness should be bigger than min_edge_thickness {}".format(max_edge_thickness)

        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Binary(1)  # 隣接行列を指す
        self.penalty_value = -10.0  # efficiencyに関するペナルティの値
        self.penalty_cross_node_value = -10.0  # 条件ノードが連結していないときの交差ノード数の

    def evaluate(self, solution):
        solution.objectives[:] = [self.objective(solution)]

    def convert_var_to_arg(self, vars):
        nodes_pos = np.array(vars[0:self.gene_node_pos_num])
        nodes_pos = nodes_pos.reshape([int(self.gene_node_pos_num / 2), 2])
        edges_thickness = vars[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num]
        adj_element = vars[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num]
        adj_element = np.array(adj_element).astype(np.int).squeeze()
        return nodes_pos, edges_thickness, adj_element

    def objective(self, solution):
        # TODO condition edges_indicesの中身は左の方が右よりも小さいということをassertする
        return self.calculate_efficiency(*self.convert_var_to_arg(solution.variables))

    def calculate_trigger(self, nodes_pos, edges_indices):  # barfemをかけるか，かけずにスコアを返すかの判断を行う関数
        # 条件ノードが含まれている部分グラフを抽出
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)
        condition_node_list = self.input_nodes + self.output_nodes + self.frozen_nodes

        trigger = 0  # 条件ノードが全て接続するグラフが存在するとき，トリガーを発動する
        for c in nx.connected_components(G):
            sg = G.subgraph(c)  # 部分グラフ
            if set(condition_node_list) <= set(sg.nodes):  # 条件ノードが全て含まれているグラフを抽出する
                edges_indices = np.array(sg.edges)
                trigger = 1
                break
        return trigger

    def return_score(self, nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix):
        trigger = self.calculate_trigger(nodes_pos, edges_indices)
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            efficiency = self.penalty_value
        else:
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
            # 傾きが一致するものをグループ分けし，エッジ分割を行う．
            processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

            if not cross_fix:  # 交差を許容しない場合
                if check_cross_graph(processed_nodes_pos, processed_edges_indices):
                    efficiency = self.penalty_value
                else:
                    # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
                    input_nodes, output_nodes, frozen_nodes, processed_edges_thickness \
                        = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                                    self.condition_edges_indices, self.condition_edges_thickness,
                                                                    processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

                    # efficiencyを計算する
                    input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
                    displacement = barfem(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                          self.input_vectors, frozen_nodes, mode='displacement')

                    efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)
            else:
                # 交差処理を行う
                processed_nodes_pos, processed_edges_indices, processed_edges_thickness =\
                    seperate_cross_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

                # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
                processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                    preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

                # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
                input_nodes, output_nodes, frozen_nodes, processed_edges_thickness \
                    = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                                self.condition_edges_indices, self.condition_edges_thickness,
                                                                processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

                # efficiencyを計算する
                input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
                displacement = barfem(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                      self.input_vectors, frozen_nodes, mode='displacement', slender=True)

                efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

            if np_save_dir:  # グラフの画像を保存する
                os.makedirs(np_save_dir, exist_ok=True)
                render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=False)
                save_graph_info_npy(np_save_dir, processed_nodes_pos, input_nodes, self.input_vectors,
                                    output_nodes, self.output_vectors, frozen_nodes,
                                    processed_edges_indices, processed_edges_thickness)

        return float(efficiency)

    def calculate_efficiency(self, gene_nodes_pos, gene_edges_thickness, gene_adj_element, np_save_dir=False, cross_fix=False):
        # make edge_indices
        edges_indices = make_adj_triu_matrix(gene_adj_element, self.node_num, self.condition_edges_indices)

        # make nodes_pos
        nodes_pos = np.concatenate([self.condition_nodes_pos, gene_nodes_pos])

        # make edge_thickness
        edges_thickness = make_edge_thick_triu_matrix(gene_edges_thickness, self.node_num,
                                                      self.condition_edges_indices, self.condition_edges_thickness, edges_indices)
        return self.return_score(nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix)


class IncrementalNodeIncrease_GA(Barfem_GA):
    def __init__(self, free_node_num, fix_node_num, max_edge_thickness=1.0, min_edge_thickness=0.5, condition_edge_thickness=0.5):
        super(IncrementalNodeIncrease_GA, self).__init__(free_node_num, fix_node_num, max_edge_thickness, min_edge_thickness, condition_edge_thickness)
        self.input_output_nodes_pos, self.input_nodes, self.input_vectors, self.output_nodes, \
            self.output_vectors, self.frozen_nodes, self.condition_edges_indices, self.condition_edges_thickness\
            = make_main_node_edge_info(*condition_only_input_output())
        self.gene_node_pos_num = self.free_node_num * 2 + self.fix_node_num
        self.frozen_nodes = np.arange(self.input_output_node_num, self.input_output_node_num + self.fix_node_num).tolist()
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1)
        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Binary(1)  # 隣接行列を指す

    def convert_var_to_arg(self, vars):
        free_nodes_pos = np.array(vars[0:self.free_node_num * 2])
        free_nodes_pos = free_nodes_pos.reshape([self.free_node_num, 2])
        fix_nodes_pos = vars[self.free_node_num * 2:self.gene_node_pos_num]
        edges_thickness = vars[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num]
        adj_element = vars[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num]
        adj_element = np.array(adj_element).astype(np.int).squeeze()
        return free_nodes_pos, fix_nodes_pos, edges_thickness, adj_element
    """
    def calculate_trigger(self, nodes_pos, edges_indices):  # 全てのノードが一つのグラフに収まる時にのみbarfemを適用するようにする
        # 条件ノードが含まれている部分グラフを抽出
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)
        condition_node_list = self.input_nodes + self.output_nodes + self.frozen_nodes

        trigger = 0  # 条件ノードが全て接続するグラフが存在するとき，トリガーを発動する
        for c in nx.connected_components(G):
            sg = G.subgraph(c)  # 部分グラフ
            if set(condition_node_list) <= set(sg.nodes):  # 条件ノードが全て含まれているグラフを抽出する
                if set(list(range(self.node_num))) <= set(sg.nodes):
                    edges_indices = np.array(sg.edges)
                    trigger = 1
                    break
        return trigger
    """

    def calculate_efficiency(self, gene_nodes_pos, gene_fix_nodes_pos, gene_edges_thickness, gene_adj_element, np_save_dir=False, cross_fix=False):
        # make nodes_pos
        gene_fix_nodes_pos = np.array(gene_fix_nodes_pos).reshape([self.fix_node_num, 1])
        gene_fix_nodes_pos = np.concatenate([gene_fix_nodes_pos, np.zeros((self.fix_node_num, 1))], 1)
        nodes_pos = np.concatenate([self.input_output_nodes_pos, gene_fix_nodes_pos, gene_nodes_pos])

        # make edge_indices
        self.condition_nodes_pos = np.concatenate([self.input_output_nodes_pos, gene_fix_nodes_pos])
        frozen_sort_nodes = np.array(self.frozen_nodes)[np.argsort(nodes_pos[self.frozen_nodes][:, 0])]
        frozen_edge_indices_1 = frozen_sort_nodes[1:]
        frozen_edge_indices_2 = frozen_sort_nodes[:-1]
        self.condition_edges_indices = np.stack([frozen_edge_indices_1, frozen_edge_indices_2], 1)  # 固定ノード間のedge_indicesを作成
        self.condition_edges_indices = np.sort(np.array(self.condition_edges_indices), axis=1)  # [[1,2],[2,3]]とか，sortされた状態になっているようにする
        edges_indices = make_adj_triu_matrix(gene_adj_element, self.node_num, self.condition_edges_indices)
        self.condition_edges_thickness = np.ones(self.condition_edges_indices.shape[0]) * self.condition_edge_thickness

        # make edge_thickness
        edges_thickness = make_edge_thick_triu_matrix(gene_edges_thickness, self.node_num,
                                                      self.condition_edges_indices, self.condition_edges_thickness, edges_indices)
        return self.return_score(nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix)


class ConstraintIncrementalNodeIncrease_GA(IncrementalNodeIncrease_GA):
    def __init__(self, free_node_num, fix_node_num, max_edge_thickness=1.0, min_edge_thickness=0.5, condition_edge_thickness=0.5, distance_threshold=0.05):
        super(ConstraintIncrementalNodeIncrease_GA, self).__init__(free_node_num, fix_node_num, max_edge_thickness, min_edge_thickness, condition_edge_thickness)
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1, 2)
        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Binary(1)  # 隣接行列を指す
        self.constraints[:] = "<=0"
        self.penalty_constraint_value = 10000
        self.distance_threshold = distance_threshold

    def evaluate(self, solution):
        [efficiency, cross_point_number, erased_node_num] = self.objective(solution)
        solution.objectives[:] = [efficiency]
        solution.constraints[:] = [cross_point_number, erased_node_num]

    def preprocess_node_joint_in_distance_threshold(self, nodes_pos):
        indexes = np.arange(nodes_pos.shape[0]).reshape((nodes_pos.shape[0], 1))
        nodes_pos_info = np.concatenate([nodes_pos, indexes], axis=1)
        while True:
            ref_nodes_pos_info = nodes_pos_info[0]
            ref_nodes_pos = ref_nodes_pos_info[:2]
            nodes_pos_info = np.delete(nodes_pos_info, 0, 0)
            lengths = [calc_length(i[0], i[1], ref_nodes_pos[0], ref_nodes_pos[1]) for i in nodes_pos_info[:, :2]]
            near_node_info_index = np.argwhere(np.array(lengths) < self.distance_threshold)
            if len(near_node_info_index) >= 1:
                same_node_pos_info_group = np.concatenate([[ref_nodes_pos_info], nodes_pos_info[near_node_info_index.squeeze()].reshape([-1, 3])])
                nodes_pos_info = np.delete(nodes_pos_info, near_node_info_index.squeeze(), 0)
                same_node_pos_info_group_indexes = same_node_pos_info_group[:, 2].astype(np.int)
                # 条件ノードが近似グループの中に存在する場合，置換するnode_posを固定ノードのnode_posにする．
                # もし複数条件ノードが存在する場合，一番indexが小さいノードのnode_posに置換する
                condition_indexes = []
                for i in same_node_pos_info_group[:, :2]:
                    target_index = np.argwhere((self.condition_nodes_pos[:, 0] == i[0]) & (self.condition_nodes_pos[:, 1] == i[1]))
                    if target_index.size != 0:
                        condition_indexes.append(target_index)
                if len(condition_indexes) != 0:
                    ref_nodes_pos = self.condition_nodes_pos[np.min(condition_indexes)]
                nodes_pos[same_node_pos_info_group_indexes] = ref_nodes_pos
            if nodes_pos_info.shape[0] == 0:
                break
        return nodes_pos

    def return_score(self, nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix):
        trigger = self.calculate_trigger(nodes_pos, edges_indices)
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            efficiency = self.penalty_value
            cross_point_num = self.penalty_constraint_value
            erased_node_num = self.penalty_constraint_value
        else:
            if self.distance_threshold:  # 近いノードを同一のノードとして処理する
                nodes_pos = self.preprocess_node_joint_in_distance_threshold(nodes_pos)
                condition_nodes_pos = []
                for i in nodes_pos:
                    if np.any((self.condition_nodes_pos[:, 0] == i[0]) & (self.condition_nodes_pos[:, 1] == i[1])):
                        condition_nodes_pos.append(i)
                self.condition_nodes_pos = np.array(nodes_pos[0: self.condition_nodes_pos.shape[0]])

            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
            # 傾きが一致するものをグループ分けし，エッジ分割を行う．
            processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

            cross_point_num = count_cross_points(processed_nodes_pos, processed_edges_indices)

            """ IncrementalNodeIncrease_GAとの正確な対照実験の時の為の処理
            if cross_point_num != 0:
                efficiency = self.penalty_value
            else:
                """
            # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
            input_nodes, output_nodes, frozen_nodes, processed_edges_thickness\
                = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                            self.condition_edges_indices, self.condition_edges_thickness,
                                                            processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # efficiencyを計算する
            input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
            displacement = barfem(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                  self.input_vectors, frozen_nodes, mode='displacement')
            efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

            erased_node_num = self.node_num - processed_nodes_pos.shape[0]

            if np_save_dir:  # グラフの画像を保存する
                os.makedirs(np_save_dir, exist_ok=True)
                render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=True)
                save_graph_info_npy(np_save_dir, processed_nodes_pos, input_nodes, self.input_vectors,
                                    output_nodes, self.output_vectors, frozen_nodes,
                                    processed_edges_indices, processed_edges_thickness)

        return float(efficiency), cross_point_num, erased_node_num


class FixnodeconstIncrementalNodeIncrease_GA(Barfem_GA):
    def __init__(self, free_node_num, fix_node_num, max_edge_thickness=0.015, min_edge_thickness=0.005, condition_edge_thickness=0.01, distance_threshold=0.05):
        super(FixnodeconstIncrementalNodeIncrease_GA, self).__init__(free_node_num, fix_node_num, max_edge_thickness, min_edge_thickness, condition_edge_thickness)
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1, 2)
        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[1::2] = Real(distance_threshold + 0.01, 1)  # ノードのy座標を固定部から離す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Binary(1)  # 隣接行列を指す
        self.constraints[:] = "<=0"
        self.penalty_constraint_value = 10000
        self.distance_threshold = distance_threshold
        assert distance_threshold < float(1 / 8), "固定ノードが同じノードとして処理される"

    def evaluate(self, solution):
        [efficiency, cross_point_number, erased_node_num] = self.objective(solution)
        solution.objectives[:] = [efficiency]
        solution.constraints[:] = [cross_point_number, erased_node_num]

    def preprocess_node_joint_in_distance_threshold(self, nodes_pos):
        indexes = np.arange(nodes_pos.shape[0]).reshape((nodes_pos.shape[0], 1))
        nodes_pos_info = np.concatenate([nodes_pos, indexes], axis=1)
        while True:
            ref_nodes_pos_info = nodes_pos_info[0]
            ref_nodes_pos = ref_nodes_pos_info[:2]
            nodes_pos_info = np.delete(nodes_pos_info, 0, 0)
            lengths = [calc_length(i[0], i[1], ref_nodes_pos[0], ref_nodes_pos[1]) for i in nodes_pos_info[:, :2]]
            near_node_info_index = np.argwhere(np.array(lengths) < self.distance_threshold)
            if len(near_node_info_index) >= 1:
                same_node_pos_info_group = np.concatenate([[ref_nodes_pos_info], nodes_pos_info[near_node_info_index.squeeze()].reshape([-1, 3])])
                nodes_pos_info = np.delete(nodes_pos_info, near_node_info_index.squeeze(), 0)
                same_node_pos_info_group_indexes = same_node_pos_info_group[:, 2].astype(np.int)
                # 条件ノードが近似グループの中に存在する場合，置換するnode_posを固定ノードのnode_posにする．
                # もし複数条件ノードが存在する場合，一番indexが小さいノードのnode_posに置換する
                condition_indexes = []
                for i in same_node_pos_info_group[:, :2]:
                    target_index = np.argwhere((self.condition_nodes_pos[:, 0] == i[0]) & (self.condition_nodes_pos[:, 1] == i[1]))
                    if target_index.size != 0:
                        condition_indexes.append(target_index)
                if len(condition_indexes) != 0:
                    ref_nodes_pos = self.condition_nodes_pos[np.min(condition_indexes)]
                nodes_pos[same_node_pos_info_group_indexes] = ref_nodes_pos
            if nodes_pos_info.shape[0] == 0:
                break
        return nodes_pos

    def return_score(self, nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix):
        trigger = self.calculate_trigger(nodes_pos, edges_indices)
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            efficiency = self.penalty_value
            cross_point_num = self.penalty_constraint_value
            erased_node_num = self.penalty_constraint_value
        else:
            if self.distance_threshold:  # 近いノードを同一のノードとして処理する
                nodes_pos = self.preprocess_node_joint_in_distance_threshold(nodes_pos)

            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
            # 傾きが一致するものをグループ分けし，エッジ分割を行う．
            processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

            cross_point_num = count_cross_points(processed_nodes_pos, processed_edges_indices)

            # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
            input_nodes, output_nodes, frozen_nodes, processed_edges_thickness\
                = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                            self.condition_edges_indices, self.condition_edges_thickness,
                                                            processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # efficiencyを計算する
            input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
            displacement = barfem(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                  self.input_vectors, frozen_nodes, mode='displacement')
            efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

            erased_node_num = self.node_num - processed_nodes_pos.shape[0]

            if np_save_dir:  # グラフの画像を保存する
                os.makedirs(np_save_dir, exist_ok=True)
                render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=True)
                save_graph_info_npy(np_save_dir, processed_nodes_pos, input_nodes, self.input_vectors,
                                    output_nodes, self.output_vectors, frozen_nodes,
                                    processed_edges_indices, processed_edges_thickness)

        return float(efficiency), cross_point_num, erased_node_num


class VolumeConstraint_GA(FixnodeconstIncrementalNodeIncrease_GA):
    def __init__(self, free_node_num, fix_node_num, max_edge_thickness=0.1, min_edge_thickness=0.005, condition_edge_thickness=0.01, distance_threshold=0.05, constraint_volume=0.3):
        super(VolumeConstraint_GA, self).__init__(free_node_num, fix_node_num, max_edge_thickness, min_edge_thickness, condition_edge_thickness)
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1, 5)
        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[1::2] = Real(distance_threshold + 0.01, 1)  # ノードのy座標を固定部から離す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Binary(1)  # 隣接行列を指す
        self.constraints[:] = "<=0"
        self.constraints[4] = ">=" + str(constraint_volume)

    def evaluate(self, solution):
        [efficiency, cross_point_number, erased_node_num, invalid_edge_num, invalid_node_num, volume] = self.objective(solution)
        solution.objectives[:] = [efficiency]
        solution.constraints[:] = [cross_point_number, erased_node_num, invalid_edge_num, invalid_node_num, volume]

    def count_invalid_edge(self, nodes_pos, edges_indices, input_nodes, frozen_nodes, free_nodes):
        # 無意味なエッジの数を数える関数
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)

        invalid_edge_num = 0

        # 入力ノード-自由ノード-固定ノードや固定ノード-自由ノード-固定ノードなどの組み合わせを検知
        input_frozen_nodes = input_nodes + frozen_nodes
        for free_node in free_nodes:
            adjacent_nodes = list(G.neighbors(free_node))
            if set(adjacent_nodes) <= set(input_frozen_nodes):
                invalid_edge_num += len(adjacent_nodes)

        # 入力ノード-固定ノードの組み合わせを検知
        for input_node in input_nodes:
            adjacent_nodes = list(G.neighbors(input_node))
            l1_l2_and = list(set(adjacent_nodes) & set(frozen_nodes))
            invalid_edge_num += len(l1_l2_and)

        return invalid_edge_num

    def count_invalid_node(self, nodes_pos, edges_indices, free_nodes):
        # 自由ノードの中で，無駄な（次数が1しかないもの）ノードの数を数えるもの
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)

        free_node_degree = [G.degree(free_node) for free_node in free_nodes]
        penalty_free_node_num = np.count_nonzero(np.array(free_node_degree) < 2)
        return penalty_free_node_num

    def return_score(self, nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix):
        trigger = self.calculate_trigger(nodes_pos, edges_indices)
        volume = calc_volume(nodes_pos, edges_indices, edges_thickness)
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            efficiency = self.penalty_value
            cross_point_num = self.penalty_constraint_value
            erased_node_num = self.penalty_constraint_value
            invalid_edge_num = self.penalty_constraint_value
            invalid_node_num = self.penalty_constraint_value
        else:
            if self.distance_threshold:  # 近いノードを同一のノードとして処理する
                nodes_pos = self.preprocess_node_joint_in_distance_threshold(nodes_pos)

            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
            # 傾きが一致するものをグループ分けし，エッジ分割を行う．
            processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

            cross_point_num = count_cross_points(processed_nodes_pos, processed_edges_indices)

            # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
            input_nodes, output_nodes, frozen_nodes, processed_edges_thickness\
                = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                            self.condition_edges_indices, self.condition_edges_thickness,
                                                            processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # efficiencyを計算する
            input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
            displacement = barfem(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                  self.input_vectors, frozen_nodes, mode='displacement')
            efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

            erased_node_num = self.node_num - processed_nodes_pos.shape[0]

            condition_nodes = input_nodes + output_nodes + frozen_nodes
            free_nodes = list(set(list(range(processed_nodes_pos.shape[0]))) ^ set(condition_nodes))
            invalid_edge_num = self.count_invalid_edge(processed_nodes_pos, processed_edges_indices, input_nodes, frozen_nodes, free_nodes)
            invalid_node_num = self.count_invalid_node(nodes_pos, edges_indices, free_nodes)

            if np_save_dir:  # グラフの画像を保存する
                os.makedirs(np_save_dir, exist_ok=True)
                render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=True)
                save_graph_info_npy(np_save_dir, processed_nodes_pos, input_nodes, self.input_vectors,
                                    output_nodes, self.output_vectors, frozen_nodes,
                                    processed_edges_indices, processed_edges_thickness)

        return float(efficiency), cross_point_num, erased_node_num, invalid_edge_num, invalid_node_num, volume


class Ansys_GA(FixnodeconstIncrementalNodeIncrease_GA):
    def __init__(self, mapdl, free_node_num, fix_node_num, max_edge_thickness=0.015, min_edge_thickness=0.005, condition_edge_thickness=0.01, distance_threshold=0.05):
        super(Ansys_GA, self).__init__(free_node_num, fix_node_num, max_edge_thickness, min_edge_thickness, condition_edge_thickness, distance_threshold)
        self.mapdl = mapdl

    def return_score(self, nodes_pos, edges_indices, edges_thickness, np_save_dir, cross_fix):
        trigger = self.calculate_trigger(nodes_pos, edges_indices)
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            efficiency = self.penalty_value
            cross_point_num = self.penalty_constraint_value
            erased_node_num = self.penalty_constraint_value
        else:
            if self.distance_threshold:  # 近いノードを同一のノードとして処理する
                nodes_pos = self.preprocess_node_joint_in_distance_threshold(nodes_pos)

            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
            # 傾きが一致するものをグループ分けし，エッジ分割を行う．
            processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
            processed_nodes_pos, processed_edges_indices, processed_edges_thickness = \
                preprocess_graph_info(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)

            cross_point_num = count_cross_points(processed_nodes_pos, processed_edges_indices)

            # 条件ノード部分の修正を行う．（太さを指定通りのものに戻す）
            input_nodes, output_nodes, frozen_nodes, processed_edges_thickness\
                = conprocess_seperate_edge_indice_procedure(self.input_nodes, self.output_nodes, self.frozen_nodes, self.condition_nodes_pos,
                                                            self.condition_edges_indices, self.condition_edges_thickness,
                                                            processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
            # efficiencyを計算する
            input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices = remove_node_which_nontouchable_in_edge_indices(input_nodes, output_nodes, frozen_nodes, processed_nodes_pos, processed_edges_indices)
            print("開始")
            displacement = barfem_mapdl(self.mapdl, processed_nodes_pos, processed_edges_indices, processed_edges_thickness, input_nodes,
                                        self.input_vectors, frozen_nodes)
            print("終了")
            efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

            erased_node_num = self.node_num - processed_nodes_pos.shape[0]

            if np_save_dir:  # グラフの画像を保存する
                os.makedirs(np_save_dir, exist_ok=True)
                render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=True)
                save_graph_info_npy(np_save_dir, processed_nodes_pos, input_nodes, self.input_vectors,
                                    output_nodes, self.output_vectors, frozen_nodes,
                                    processed_edges_indices, processed_edges_thickness)

        return float(efficiency), cross_point_num, erased_node_num
