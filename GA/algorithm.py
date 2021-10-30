from platypus import NSGAII, default_variator
from tqdm import tqdm
from GA.GA_class import *
from tools.graph import *
import numpy as np
import os
from GA.utils import *
import pickle


class Customized_NSGAII(NSGAII):
    def __init__(self, problem, prior_problem, gene_path,
                 population_size=100,
                 variator=None,
                 archive=None,
                 init_parents=None,
                 **kwargs):
        super(Customized_NSGAII, self).__init__(problem,
                                                population_size=population_size,
                                                variator=variator,
                                                archive=archive,
                                                **kwargs)
        self.problem = problem
        self.prior_problem = prior_problem
        self.gene_path = gene_path

    def initialize(self):
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
        # ここで，populationの内の半分を変える
        with open(self.gene_path, 'rb') as f:
            genes = pickle.load(f)

        if self.problem.free_node_num > self.prior_problem.free_node_num:  # 自由ノード拡張時
            inherit_function = self.make_inherit_genes_with_increasing_free_node
        else:  # 固定ノード拡張時
            inherit_function = self.make_inherit_genes_with_increasing_fix_node

        gene_solutions = []
        print("引継ぎ開始")
        for i in tqdm(range(len(genes))):
            gene_solutions.append(inherit_function(genes[i], self.population[0:len(genes)][i]))

        self.population[0:len(genes)] = gene_solutions

        self.evaluate_all(self.population)

        if self.archive is not None:
            self.archive += self.population

        if self.variator is None:
            self.variator = default_variator(self.problem)

    def make_inherit_genes_with_increasing_free_node(self, gene, solution):
        pro_free_node_num = self.problem.free_node_num
        pro_fix_node_num = self.problem.fix_node_num
        input_output_node_num = self.problem.input_output_node_num
        prior_free_node_num = pro_free_node_num - 1
        prior_fix_node_num = pro_fix_node_num

        # inherit free node
        random_free_node = np.random.rand(2).tolist()
        gene_node_pos = gene[0:prior_free_node_num * 2 + prior_fix_node_num]
        gene_node_pos[prior_free_node_num * 2:prior_free_node_num * 2] = random_free_node  # add random free node
        solution.variables[0:pro_free_node_num * 2 + pro_fix_node_num] = gene_node_pos

        # inherit edge_indices
        prior_node_num = (2 + prior_free_node_num + prior_fix_node_num)
        prior_gene_edge_indices_num = int(prior_node_num * (prior_node_num - 1) / 2)

        random_edge_indices = np.zeros(prior_node_num, dtype=bool)
        random_edge_indices[np.random.randint(0, len(random_edge_indices))] = True
        random_edge_indices = random_edge_indices.reshape((-1, 1)).tolist()

        gene_edge_indices = gene[-prior_gene_edge_indices_num:]

        new_gene_edge_indices = add_edge_indices_to_gene_edge_indices(gene_edge_indices, random_edge_indices, prior_node_num)
        pro_node_num = (input_output_node_num + pro_free_node_num + pro_fix_node_num)
        pro_gene_edge_indices_num = int(pro_node_num * (pro_node_num - 1) / 2)
        solution.variables[-pro_gene_edge_indices_num:] = new_gene_edge_indices

        # inherit edge_thickness
        prior_gene_node_pos_num = prior_free_node_num * 2 + prior_fix_node_num
        prior_gene_edge_thickness_num = int(prior_node_num * (prior_node_num - 1) / 2)
        gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
        additional_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
        new_gene_edge_thickness = self.add_free_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, additional_edge_thickness, prior_node_num)
        solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness

        return solution

    def make_inherit_genes_with_increasing_fix_node(self, gene, solution):
        pro_free_node_num = self.problem.free_node_num
        pro_fix_node_num = self.problem.fix_node_num
        pro_node_num = 2 + pro_free_node_num + pro_fix_node_num
        input_output_node_num = self.problem.input_output_node_num
        prior_free_node_num = pro_free_node_num
        prior_fix_node_num = pro_fix_node_num - 1
        prior_node_num = 2 + prior_free_node_num + prior_fix_node_num

        # inherit fix node
        random_fix_node = np.random.rand(1).tolist()
        gene_node_pos = gene[0:prior_free_node_num * 2 + prior_fix_node_num]
        gene_node_pos.extend(random_fix_node)  # add random fix node
        solution.variables[0:pro_free_node_num * 2 + pro_fix_node_num] = gene_node_pos

        # inherit edge_indices
        prior_gene_edge_indices_num = int(prior_node_num * (prior_node_num - 1) / 2)
        gene_edge_indices = gene[-prior_gene_edge_indices_num:]
        adj_matrix = np.zeros((prior_node_num, prior_node_num), dtype=bool)
        adj_matrix[np.triu_indices(prior_node_num, 1)] = np.array(gene_edge_indices).squeeze()
        adj_matrix = np.insert(adj_matrix, input_output_node_num + prior_fix_node_num, np.zeros((1, prior_node_num), dtype=bool), axis=0)
        adj_matrix = np.insert(adj_matrix, input_output_node_num + prior_fix_node_num, np.zeros((1, pro_node_num), dtype=bool), axis=1)
        new_gene_edge_indices = adj_matrix[np.triu_indices(pro_node_num, 1)].reshape((-1, 1)).tolist()
        pro_gene_edge_indices_num = int(pro_node_num * (pro_node_num - 1) / 2)
        solution.variables[-pro_gene_edge_indices_num:] = new_gene_edge_indices

        # inherit edge_thickness
        prior_gene_node_pos_num = prior_free_node_num * 2 + prior_fix_node_num
        prior_gene_edge_thickness_num = int(prior_node_num * (prior_node_num - 1) / 2)
        gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
        new_gene_edge_thickness = self.add_fix_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, prior_node_num, input_output_node_num, prior_fix_node_num, prior_node_num, pro_node_num)
        solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness

        return solution

    def add_free_node_edge_thickness_to_gene_edge_thickness(self, gene_edge_thickness, additional_edge_thickness, gene_node_num):
        """gene_edge_thicknessに対し，新edge_thicknessを適切に追加する関数
        """
        adj_matrix = np.zeros((gene_node_num, gene_node_num))
        adj_matrix[np.triu_indices(gene_node_num, 1)] = np.array(gene_edge_thickness)
        additional_edge_thickness = np.reshape(additional_edge_thickness, [gene_node_num, 1])
        new_adj_matrix = np.concatenate([adj_matrix, additional_edge_thickness], axis=1)
        new_adj_matrix = np.concatenate([new_adj_matrix, np.zeros((1, gene_node_num + 1))], axis=0)
        new_gene_edge_thickness = new_adj_matrix[np.triu_indices(gene_node_num + 1, 1)].tolist()
        return new_gene_edge_thickness

    def add_fix_node_edge_thickness_to_gene_edge_thickness(self, gene_edge_thickness, gene_node_num, input_output_node_num, prior_fix_node_num, prior_node_num, pro_node_num):
        """gene_edge_thicknessに対し，新edge_thicknessを適切に追加する関数
        """
        adj_matrix = np.zeros((gene_node_num, gene_node_num))
        adj_matrix[np.triu_indices(gene_node_num, 1)] = np.array(gene_edge_thickness)
        add_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
        adj_matrix = np.insert(adj_matrix, input_output_node_num + prior_fix_node_num, add_edge_thickness, axis=0)
        add_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(pro_node_num) + self.problem.min_edge_thickness
        adj_matrix = np.insert(adj_matrix, input_output_node_num + prior_fix_node_num, add_edge_thickness, axis=1)
        new_gene_edge_thickness = adj_matrix[np.triu_indices(gene_node_num + 1, 1)].tolist()
        return new_gene_edge_thickness


class FixNode_NSGAII(Customized_NSGAII):
    def __init__(self, problem, prior_problem, gene_path,
                 population_size=100,
                 variator=None,
                 archive=None,
                 init_parents=None,
                 **kwargs):
        super(FixNode_NSGAII, self).__init__(problem, prior_problem, gene_path,
                                             population_size=population_size,
                                             variator=variator,
                                             archive=archive,
                                             **kwargs)

    def make_inherit_genes_with_increasing_free_node(self, gene, solution):
        prior_node_num = self.prior_problem.node_num
        pro_node_num = self.problem.node_num

        gene_feasible_condition = gene.feasible
        gene_result = self.prior_problem.objective(gene)
        gene_efficiency, gene_erased_node_num = gene_result["efficiency"], gene_result["erased_node_num"]
        gene = gene.variables

        child_nodes_pos, child_edges_indices, child_gene_edges_thickness = self.prior_problem.get_graph_info_from_genes(gene)
        child_edge_points = np.array([np.stack([child_nodes_pos[edges_indice[0]], child_nodes_pos[edges_indice[1]]]) for edges_indice in child_edges_indices])

        while True:  # 引き継いだgeneが同じ条件になる為に何度も繰り返す
            # inherit free node and edge_indices
            while True:
                while True:
                    random_free_node_pos = np.random.rand(2)
                    distances = np.linalg.norm(child_nodes_pos - random_free_node_pos, axis=1)
                    if not np.any(distances < self.prior_problem.distance_threshold):  # 新たに追加するノードが，他のノードとくっつかないようにする
                        break
                sort_index = np.argsort(distances)
                fix_nodes_num = np.ones(prior_node_num, dtype=np.int) * (self.problem.node_num - 1)
                candidate_edges_indices = np.stack([sort_index, fix_nodes_num], axis=1)
                candidate_edge_points = np.array([np.stack([child_nodes_pos[edges_indice[0]], random_free_node_pos]) for edges_indice in candidate_edges_indices])
                for index, edge_points in enumerate(candidate_edge_points):  # それぞれの候補エッジがどの既存のエッジとも交差しないかどうかをチェックする
                    cross_check = np.all([not calc_cross_point(edge_points[0], edge_points[1], i[0], i[1])[0] for i in child_edge_points])
                    if cross_check:
                        chosen_index = index
                        break
                if cross_check:
                    break
            add_edge_indices = candidate_edges_indices[chosen_index].reshape((-1, 2))
            new_gene_edges_indices = np.concatenate([child_edges_indices, add_edge_indices])
            new_gene_edges_indices = revert_edge_indices_to_binary(new_gene_edges_indices, pro_node_num)
            solution.variables[-self.problem.gene_edge_indices_num:] = new_gene_edges_indices
            gene_node_pos = gene[0:self.prior_problem.gene_node_pos_num]
            gene_node_pos[self.prior_problem.gene_node_pos_num:self.prior_problem.gene_node_pos_num] = random_free_node_pos.tolist()  # add random free node
            solution.variables[0:self.problem.gene_node_pos_num] = gene_node_pos
            # inherit edge_thickness
            prior_gene_node_pos_num = self.prior_problem.gene_node_pos_num
            prior_gene_edge_thickness_num = self.prior_problem.gene_edge_thickness_num
            gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
            additional_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
            new_gene_edge_thickness = self.add_free_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, additional_edge_thickness, prior_node_num)
            solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness

            solution.evaluated = False
            self.evaluate_all([solution])
            if solution.feasible == gene_feasible_condition:
                if gene_feasible_condition:
                    solution_result = self.problem.objective(solution)
                    solution_efficiency, solution_erased_node_num = solution_result["efficiency"], solution_result["erased_node_num"]
                    if np.isclose(gene_efficiency, solution_efficiency) and solution_erased_node_num == gene_erased_node_num:  # feasible解においてノード数が消えた状態で引き継がれることを阻止する為
                        break
                else:
                    break

        return solution


class FixNode_add_middle_point_NSGAII(Customized_NSGAII):
    def __init__(self, problem, prior_problem, gene_path,
                 population_size=100,
                 variator=None,
                 archive=None,
                 init_parents=None,
                 **kwargs):
        super(FixNode_add_middle_point_NSGAII, self).__init__(problem, prior_problem, gene_path,
                                                              population_size=population_size,
                                                              variator=variator,
                                                              archive=archive,
                                                              **kwargs)
        self.frozen_nodes = self.prior_problem.frozen_nodes  # priorにしていることに特に意味はない

    def check_equal_to_condition_edge(self, target_edge_indice):
        if ((target_edge_indice[0] in self.frozen_nodes) & (target_edge_indice[1] in self.frozen_nodes)):
            return True
        return False

    def check_not_condition_for_selected_edge(self, child_nodes_pos, child_edges_indices):
        edge_points = np.array([np.stack([child_nodes_pos[edges_indice[0]], child_nodes_pos[edges_indice[1]]]) for edges_indice in child_edges_indices])
        lengths = np.array([calc_length(i[0][0], i[0][1], i[1][0], i[1][1]) for i in edge_points])
        lengths_condition = lengths < (self.problem.distance_threshold * 2)  # 長さが2d未満であるものをTrueとする．

        y_lower_bound = self.problem.distance_threshold
        vectors = edge_points[:, 0] - edge_points[:, 1]
        # ノード1側がself.problem.distance_thresholdよりも下にあり，ノード2が上にある場合
        node1_under_d_cond = (edge_points[:, 0, 1] < y_lower_bound) & (edge_points[:, 1, 1] >= y_lower_bound)  # ノード1が下側のindex．最終的には条件違反のものをTrueにする．
        dd_L = ((y_lower_bound - edge_points[node1_under_d_cond, 0, 1]) / -vectors[node1_under_d_cond, 1])
        d_L = self.problem.distance_threshold / lengths[node1_under_d_cond]
        restricted_index = (((1 - dd_L) < d_L) & (dd_L >= d_L))
        node_info = node1_under_d_cond[node1_under_d_cond].copy()  # copyしないと上手く行かない
        node_info[~restricted_index] = False  # node1_under_d_condの内，中間エッジが設置可能なものに対してFalseと置き換える
        node1_under_d_cond[node1_under_d_cond] = node_info  # node1_under_d_condの内，中間エッジが設置可能なものに対してFalseと置き換える

        # ノード2側がself.problem.distance_thresholdよりも下にあり，ノード1が上にある場合
        node2_under_d_cond = (edge_points[:, 1, 1] < y_lower_bound) & (edge_points[:, 0, 1] >= y_lower_bound)  # ノード2が下側のindex．最終的には条件違反のものをTrueにする．
        dd_L = ((y_lower_bound - edge_points[node2_under_d_cond, 1, 1]) / vectors[node2_under_d_cond, 1])
        d_L = self.problem.distance_threshold / lengths[node2_under_d_cond]
        restricted_index = (((1 - dd_L) < d_L) & (dd_L >= d_L))
        node_info = node2_under_d_cond[node2_under_d_cond].copy()  # copyしないと上手く行かない
        node_info[~restricted_index] = False  # node2_under_d_condの内，中間エッジが設置可能なものに対してFalseと置き換える
        node2_under_d_cond[node2_under_d_cond] = node_info

        under_d_cond = node1_under_d_cond | node2_under_d_cond

        return lengths_condition | under_d_cond

    def make_inherit_genes_with_increasing_free_node(self, gene, solution):
        add_middle_edge_time_maximum_threshold = 20  # 中間エッジを付与しようとした回数の上限値
        prior_node_num = self.prior_problem.node_num
        pro_node_num = self.problem.node_num
        gene_result = self.prior_problem.objective(gene)
        gene_efficiency, gene_erased_node_num = gene_result["efficiency"], gene_result["erased_node_num"]

        gene_feasible_condition = gene.feasible
        gene = gene.variables
        add_middle_edge_time = 0  # 中間エッジを付与しようとした回数
        while True:  # 引き継いだgeneが同じ条件になる為に何度も繰り返す
            # どのエッジの間にノードを追加するかを選ぶ
            child_nodes_pos, child_edges_indices, child_gene_edges_thickness = self.prior_problem.get_graph_info_from_genes(gene)
            remove_indexes = [not self.check_equal_to_condition_edge(i) for i in child_edges_indices]  # 条件ノード間ではないエッジのみ抽出
            edge_cond_remove_indexes = self.check_not_condition_for_selected_edge(child_nodes_pos, child_edges_indices)  # ノード間を満たすエッジのみ抽出
            candidate_indexes = np.where(remove_indexes & (~edge_cond_remove_indexes))[0]
            if len(candidate_indexes) != 0 and (add_middle_edge_time <= add_middle_edge_time_maximum_threshold):
                add_middle_edge_time += 1
                selected_edge_index = np.random.choice(candidate_indexes)
                selected_edge_indices = child_edges_indices[selected_edge_index]
                child_edges_indices = np.delete(child_edges_indices, selected_edge_index, 0)
                new_gene_edges_indices = np.concatenate([child_edges_indices, np.array([[selected_edge_indices[0], pro_node_num - 1], [selected_edge_indices[1], pro_node_num - 1]])])
                new_gene_edges_indices = revert_edge_indices_to_binary(new_gene_edges_indices, pro_node_num)
                solution.variables[-self.problem.gene_edge_indices_num:] = new_gene_edges_indices
                # inherit free node
                y_lower_bound = self.problem.distance_threshold
                selected_node1_pos = child_nodes_pos[selected_edge_indices[0]]
                selected_node2_pos = child_nodes_pos[selected_edge_indices[1]]
                vector = selected_node1_pos - selected_node2_pos
                length = calc_length(selected_node1_pos[0], selected_node1_pos[1], selected_node2_pos[0], selected_node2_pos[1])

                # 追加するノードが下限のdistance_holdよりも下に行かないようにする．なお，これは固定ノード間など，distance_hold以下のノード同士を組んだエッジが選ばれないことを前提とする．
                if (selected_node1_pos[1] >= y_lower_bound) & (selected_node2_pos[1] >= y_lower_bound):
                    hl = 1 - (self.problem.distance_threshold / length)
                    ll = self.problem.distance_threshold / length
                    r = np.random.uniform(low=ll, high=hl)
                elif selected_node1_pos[1] >= y_lower_bound:  # ノード１のみ上
                    dd_L = ((y_lower_bound - selected_node2_pos[1]) / vector[1])
                    d_L = self.problem.distance_threshold / length
                    if dd_L >= d_L:
                        r = np.random.uniform(low=dd_L, high=1 - d_L)
                    else:
                        r = np.random.uniform(low=d_L, high=1 - d_L)
                else:  # ノード２のみ上
                    dd_L = ((y_lower_bound - selected_node1_pos[1]) / -vector[1])
                    d_L = self.problem.distance_threshold / length
                    if dd_L >= d_L:
                        r = np.random.uniform(low=d_L, high=1 - dd_L)
                    else:
                        r = np.random.uniform(low=d_L, high=1 - d_L)
                add_free_node_pos = (selected_node1_pos - selected_node2_pos) * r + selected_node2_pos
                add_free_node_pos = add_free_node_pos.tolist()
                gene_node_pos = gene[0:self.prior_problem.gene_node_pos_num]
                gene_node_pos[self.prior_problem.gene_node_pos_num:self.prior_problem.gene_node_pos_num] = add_free_node_pos  # add random free node
                solution.variables[0:self.problem.gene_node_pos_num] = gene_node_pos

                # inherit edge_thickness
                add_edge_thick = child_gene_edges_thickness[selected_edge_index]  # ref_edge_thicknessは，geneではなく，
                prior_gene_node_pos_num = self.prior_problem.gene_node_pos_num
                prior_gene_edge_thickness_num = self.prior_problem.gene_edge_thickness_num
                gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
                additional_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
                additional_edge_thickness[selected_edge_indices] = add_edge_thick
                new_gene_edge_thickness = self.add_free_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, additional_edge_thickness, prior_node_num)
                solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness
            else:  # 一つも条件を満たすエッジが存在しない場合，ノードを一つ付与する
                child_edge_points = np.array([np.stack([child_nodes_pos[edges_indice[0]], child_nodes_pos[edges_indice[1]]]) for edges_indice in child_edges_indices])
                # inherit free node and edge_indices
                while True:
                    while True:
                        random_free_node_pos = np.random.rand(2)
                        distances = np.linalg.norm(child_nodes_pos - random_free_node_pos, axis=1)
                        if not np.any(distances < self.prior_problem.distance_threshold):  # 新たに追加するノードが，他のノードとくっつかないようにする
                            break
                    sort_index = np.argsort(distances)
                    fix_nodes_num = np.ones(prior_node_num, dtype=np.int) * (self.problem.node_num - 1)
                    candidate_edges_indices = np.stack([sort_index, fix_nodes_num], axis=1)
                    candidate_edge_points = np.array([np.stack([child_nodes_pos[edges_indice[0]], random_free_node_pos]) for edges_indice in candidate_edges_indices])
                    for index, edge_points in enumerate(candidate_edge_points):  # それぞれの候補エッジがどの既存のエッジとも交差しないかどうかをチェックする
                        cross_check = np.all([not calc_cross_point(edge_points[0], edge_points[1], i[0], i[1])[0] for i in child_edge_points])
                        if cross_check:
                            chosen_index = index
                            break
                    if cross_check:
                        break
                add_edge_indices = candidate_edges_indices[chosen_index].reshape((-1, 2))
                new_gene_edges_indices = np.concatenate([child_edges_indices, add_edge_indices])
                new_gene_edges_indices = revert_edge_indices_to_binary(new_gene_edges_indices, pro_node_num)
                solution.variables[-self.problem.gene_edge_indices_num:] = new_gene_edges_indices
                gene_node_pos = gene[0:self.prior_problem.gene_node_pos_num]
                gene_node_pos[self.prior_problem.gene_node_pos_num:self.prior_problem.gene_node_pos_num] = random_free_node_pos.tolist()  # add random free node
                solution.variables[0:self.problem.gene_node_pos_num] = gene_node_pos

                # inherit edge_thickness
                prior_gene_node_pos_num = self.prior_problem.gene_node_pos_num
                prior_gene_edge_thickness_num = self.prior_problem.gene_edge_thickness_num
                gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
                additional_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
                new_gene_edge_thickness = self.add_free_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, additional_edge_thickness, prior_node_num)
                solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness
            solution.evaluated = False
            self.evaluate_all([solution])
            if solution.feasible == gene_feasible_condition:
                if gene_feasible_condition:
                    solution_result = self.problem.objective(solution)
                    solution_efficiency, solution_erased_node_num = solution_result["efficiency"], solution_result["erased_node_num"]
                    if np.isclose(gene_efficiency, solution_efficiency) and solution_erased_node_num == gene_erased_node_num:  # feasible解においてノード数が消えた状態で引き継がれることを阻止する為
                        break
                else:
                    break
        return solution
