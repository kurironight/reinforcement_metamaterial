from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, PM, BitFlip
from .condition import condition
from tools.lattice_preprocess import make_main_node_edge_info
from tools.graph import preprocess_graph_info, separate_same_line_procedure, \
    conprocess_seperate_edge_indice_procedure, seperate_cross_line_procedure, calc_efficiency,\
    remove_node_which_nontouchable_in_edge_indices, render_graph
import numpy as np
from .utils import make_edge_thick_triu_matrix, make_adj_triu_matrix
import networkx as nx
import os
from FEM.bar_fem import barfem


class Barfem_GA(Problem):

    def __init__(self, node_num, max_edge_thickness=0.05, min_edge_thickness=0.01):
        self.condition_nodes_pos, self.input_nodes, self.input_vectors, self.output_nodes, \
            self.output_vectors, self.frozen_nodes, self.condition_edges_indices, self.condition_edges_thickness\
            = make_main_node_edge_info(*condition(), condition_edge_thickness=0.05)  # スレンダー比を考慮し，長さ方向に対して1/20の値の幅にした

        self.node_num = node_num
        condition_node_num = self.condition_nodes_pos.shape[0]
        assert self.node_num > condition_node_num, "node_num should be bigger than condition node num {}".format(condition_node_num)
        self.gene_node_pos_num = (node_num - condition_node_num) * 2
        self.gene_edge_thickness_num = int(node_num * (node_num - 1) / 2)
        self.gene_edge_indices_num = self.gene_edge_thickness_num
        super(Barfem_GA, self).__init__(self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num, 1)
        self.max_edge_thickness = max_edge_thickness
        self.min_edge_thickness = min_edge_thickness
        assert min_edge_thickness < self.max_edge_thickness, "max_edge_thickness should be bigger than min_edge_thickness {}".format(max_edge_thickness)

        self.directions[:] = Problem.MAXIMIZE
        self.types[0:self.gene_node_pos_num] = Real(0, 1)  # ノードの位置座標を示す
        self.types[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num] = Real(self.min_edge_thickness, self.max_edge_thickness)  # エッジの幅を示す バグが無いように0.1にする
        self.types[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num] = \
            Integer(0, 1)  # 隣接行列を指す

    def evaluate(self, solution):
        solution.objectives[:] = [self.objective(solution)]

    def convert_var_to_arg(self, vars):
        nodes_pos = np.array(vars[0:self.gene_node_pos_num])
        nodes_pos = nodes_pos.reshape([int(self.gene_node_pos_num / 2), 2])
        edges_thickness = vars[self.gene_node_pos_num:self.gene_node_pos_num + self.gene_edge_thickness_num]
        adj_element = vars[self.gene_node_pos_num + self.gene_edge_thickness_num: self.gene_node_pos_num + self.gene_edge_thickness_num + self.gene_edge_indices_num]
        return nodes_pos, edges_thickness, adj_element

    def objective(self, solution):
        # TODO condition edges_indicesの中身は左の方が右よりも小さいということをassertする
        gene_nodes_pos, gene_edges_thickness, gene_adj_element = self.convert_var_to_arg(solution.variables)
        return self.calculate_efficiency(gene_nodes_pos, gene_edges_thickness, gene_adj_element)

    def calculate_efficiency(self, gene_nodes_pos, gene_edges_thickness, gene_adj_element, np_save_dir=False, cross_fix=True):

        # make edge_indices
        edges_indices = make_adj_triu_matrix(gene_adj_element, self.node_num, self.condition_edges_indices)

        # make nodes_pos
        nodes_pos = np.concatenate([self.condition_nodes_pos, gene_nodes_pos])

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
        if trigger == 0:  # もし条件ノードが全て含まれるグラフが存在しない場合，ペナルティを発動する
            return -10.0

        # make edges_thickness
        edges_thickness = make_edge_thick_triu_matrix(gene_edges_thickness, self.node_num,
                                                      self.condition_edges_indices, self.condition_edges_thickness, edges_indices)

        # 同じノード，[1,1]などのエッジの排除，エッジのソートなどを行う
        processed_nodes_pos, processed_edges_indices, processed_edges_thickness = preprocess_graph_info(nodes_pos, edges_indices, edges_thickness)
        # 傾きが一致するものをグループ分けし，エッジ分割を行う．
        processed_edges_indices, processed_edges_thickness = separate_same_line_procedure(processed_nodes_pos, processed_edges_indices, processed_edges_thickness)
        if cross_fix:
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
                              self.input_vectors, frozen_nodes, mode='displacement')

        efficiency = calc_efficiency(input_nodes, self.input_vectors, output_nodes, self.output_vectors, displacement)

        if np_save_dir:  # グラフの画像を保存する
            render_graph(processed_nodes_pos, processed_edges_indices, processed_edges_thickness, os.path.join(np_save_dir, "image.png"), display_number=False)
            np.save(os.path.join(np_save_dir, "nodes_pos.npy"), processed_nodes_pos)
            np.save(os.path.join(np_save_dir, "edges_indices.npy"), processed_edges_indices)
            np.save(os.path.join(np_save_dir, "edges_thickness.npy"), processed_edges_thickness)

        return float(efficiency)
