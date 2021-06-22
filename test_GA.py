from GA.condition import condition
from tools.lattice_preprocess import make_main_node_edge_info
import numpy as np
from env.gym_barfem import BarFemGym
import networkx as nx
from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, PM, BitFlip
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def calculate_efficiency(gene_nodes_pos, gene_edges_thickness, gene_adj_element, np_save_path=False):
    condition_nodes_pos, input_nodes, input_vectors, output_nodes, \
        output_vectors, frozen_nodes, condition_edges_indices, condition_edges_thickness\
        = make_main_node_edge_info(*condition(), condition_edge_thickness=0.2)

    # make edge_indices
    edges_indices = make_adj_triu_matrix(gene_adj_element, node_num, condition_edges_indices)

    # make nodes_pos
    nodes_pos = np.concatenate([condition_nodes_pos, gene_nodes_pos])

    # 条件ノードが含まれている部分グラフを抽出
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes_pos)))
    G.add_edges_from(edges_indices)
    condition_node_list = input_nodes + output_nodes + frozen_nodes

    trigger = 0  # 条件ノードが全て接続するグラフが存在するとき，トリガーを発動する
    for c in nx.connected_components(G):
        sg = G.subgraph(c)  # 部分グラフ
        if set(condition_node_list) <= set(sg.nodes):  # 条件ノードが全て含まれているか
            edges_indices = np.array(sg.edges)
            trigger = 1
            break
    if trigger == 0:  # ペナルティを発動する
        return -10.0

    # make edges_thickness
    edges_thickness = make_edge_thick_triu_matrix(gene_edges_thickness, node_num, condition_edges_indices, condition_edges_thickness, edges_indices)

    env = BarFemGym(nodes_pos, input_nodes, input_vectors,
                    output_nodes, output_vectors, frozen_nodes,
                    edges_indices, edges_thickness, frozen_nodes)
    env.reset()
    efficiency = env.calculate_simulation()
    if np_save_path:
        env.render(save_path=os.path.join(np_save_path, "image.png"))
        np.save(os.path.join(np_save_path, "nodes_pos.npy"), nodes_pos)
        np.save(os.path.join(np_save_path, "edges_indices.npy"), edges_indices)
        np.save(os.path.join(np_save_path, "edges_thickness.npy"), edges_thickness)

    return float(efficiency)


def objective(vars):
    # TODO condition edges_indicesの中身は左の方が右よりも小さいということをassertする
    gene_nodes_pos, gene_edges_thickness, gene_adj_element = convert_var_to_arg(vars)
    return [calculate_efficiency(gene_nodes_pos, gene_edges_thickness, gene_adj_element)]


def make_adj_triu_matrix(adj_element, node_num, condition_edges_indices):
    """隣接情報を示す遺伝子から，edge_indicesを作成する関数
    """
    adj_matrix = np.zeros((node_num, node_num))
    adj_matrix[np.triu_indices(node_num, 1)] = adj_element

    adj_matrix[(condition_edges_indices[:, 0], condition_edges_indices[:, 1])] = 1
    edge_indices = np.stack(np.where(adj_matrix), axis=1)

    return edge_indices


def make_edge_thick_triu_matrix(gene_edges_thickness, node_num, condition_edges_indices, condition_edges_thickness, edges_indices):
    """edge_thicknessを示す遺伝子から，condition_edge_thicknessを基にedges_thicknessを作成する関数
    """
    tri = np.zeros((node_num, node_num))
    tri[np.triu_indices(node_num, 1)] = gene_edges_thickness

    tri[(condition_edges_indices[:, 0], condition_edges_indices[:, 1])] = condition_edges_thickness
    edges_thickness = tri[(edges_indices[:, 0], edges_indices[:, 1])]

    return edges_thickness


def convert_var_to_arg(vars):
    nodes_pos = np.array(vars[0:gene_node_pos_num])
    nodes_pos = nodes_pos.reshape([int(gene_node_pos_num / 2), 2])
    edges_thickness = vars[gene_node_pos_num:gene_node_pos_num + gene_edge_thickness_num]
    adj_element = vars[gene_node_pos_num + gene_edge_thickness_num: gene_node_pos_num + gene_edge_thickness_num + gene_edge_indices_num]
    return nodes_pos, edges_thickness, adj_element


node_num = 85
#parent = 20
parent = (node_num * 2 + int(node_num * (node_num - 1) / 2) * 2)  # 本来ならこれの10倍
generation = 5000
save_interval = 10

PATH = os.path.join("GA/result/parent_{}_gen_{}_2".format(parent, generation))
os.makedirs(PATH, exist_ok=True)

condition_node_num = 10
gene_node_pos_num = (node_num - condition_node_num) * 2

gene_edge_thickness_num = int(node_num * (node_num - 1) / 2)
gene_edge_indices_num = gene_edge_thickness_num

# 2変数2目的の問題
problem = Problem(gene_node_pos_num + gene_edge_thickness_num + gene_edge_indices_num, 1)

# 最小化or最大化を設定
problem.directions[:] = Problem.MAXIMIZE

# 決定変数の範囲を設定
coord_const = Real(0, 1)
edge_const = Real(0.1, 1)  # バグが無いように0.1にする
adj_constraint = Integer(0, 1)

problem.types[0:gene_node_pos_num] = coord_const
problem.types[gene_node_pos_num:gene_node_pos_num + gene_edge_thickness_num] = edge_const
problem.types[gene_node_pos_num + gene_edge_thickness_num: gene_node_pos_num + gene_edge_thickness_num + gene_edge_indices_num] = adj_constraint
problem.function = objective

algorithm = NSGAII(problem, population_size=parent,
                   variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))

history = []

for i in tqdm(range(generation)):
    algorithm.step()
    nondominated_solutions = nondominated(algorithm.result)
    efficiency_results = [s.objectives[0] for s in nondominated_solutions]
    max_efficiency = max(efficiency_results)
    history.append(max_efficiency)

    epochs = np.arange(i + 1) + 1
    result_efficiency = np.array(history)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, result_efficiency, label='efficiency')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.legend()
    ax.set_title("efficiency curve")
    plt.savefig(os.path.join(PATH, "history.png"))
    plt.close()

    if i % save_interval == 0:
        save_dir = os.path.join(PATH, str(i))
        max_index = efficiency_results.index(max_efficiency)
        max_solution = nondominated_solutions[max_index]

        vars = []
        vars.extend([coord_const.decode(i) for i in max_solution.variables[0:gene_node_pos_num]])
        vars.extend([edge_const.decode(i) for i in max_solution.variables[gene_node_pos_num:gene_node_pos_num + gene_edge_thickness_num]])
        vars.extend([adj_constraint.decode(i) for i in max_solution.variables[gene_node_pos_num + gene_edge_thickness_num: gene_node_pos_num + gene_edge_thickness_num + gene_edge_indices_num]])
        gene_nodes_pos, gene_edges_thickness, gene_adj_element = convert_var_to_arg(vars)
        calculate_efficiency(gene_nodes_pos, gene_edges_thickness, gene_adj_element, np_save_path=save_dir)

        np.save(os.path.join(save_dir, "history.npy"), history)
