from platypus import NSGAII, default_variator
from GA.GA_class import *
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

        gene_solutions = [inherit_function(gene.variables, solution) for solution, gene in zip(self.population[0:len(genes)], genes)]
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
        pro_free_node_num = self.problem.free_node_num
        pro_fix_node_num = self.problem.fix_node_num
        input_output_node_num = self.problem.input_output_node_num
        prior_free_node_num = pro_free_node_num - 1
        prior_fix_node_num = pro_fix_node_num

        # inherit free node
        random_free_node = np.random.rand(2).tolist()
        gene_node_pos = gene[0:self.prior_problem.gene_node_pos_num]
        gene_node_pos[self.prior_problem.gene_node_pos_num:self.prior_problem.gene_node_pos_num] = random_free_node  # add random free node
        solution.variables[0:self.problem.gene_node_pos_num] = gene_node_pos

        # inherit edge_indices
        prior_node_num = self.prior_problem.node_num
        prior_gene_edge_indices_num = self.prior_problem.gene_edge_indices_num

        random_edge_indices = np.zeros(prior_node_num, dtype=bool)
        random_edge_indices[np.random.randint(0, len(random_edge_indices))] = True
        random_edge_indices = random_edge_indices.reshape((-1, 1)).tolist()

        gene_edge_indices = gene[-prior_gene_edge_indices_num:]

        new_gene_edge_indices = add_edge_indices_to_gene_edge_indices(gene_edge_indices, random_edge_indices, prior_node_num)
        pro_node_num = (input_output_node_num + pro_free_node_num + pro_fix_node_num)
        pro_gene_edge_indices_num = int(pro_node_num * (pro_node_num - 1) / 2)
        solution.variables[-pro_gene_edge_indices_num:] = new_gene_edge_indices

        # inherit edge_thickness
        prior_gene_node_pos_num = self.prior_problem.gene_node_pos_num
        prior_gene_edge_thickness_num = self.prior_problem.gene_edge_thickness_num
        gene_edge_thickness = gene[prior_gene_node_pos_num:prior_gene_node_pos_num + prior_gene_edge_thickness_num]
        additional_edge_thickness = (self.problem.max_edge_thickness - self.problem.min_edge_thickness) * np.random.rand(prior_node_num) + self.problem.min_edge_thickness
        new_gene_edge_thickness = self.add_free_node_edge_thickness_to_gene_edge_thickness(gene_edge_thickness, additional_edge_thickness, prior_node_num)
        solution.variables[self.problem.gene_node_pos_num:self.problem.gene_node_pos_num + self.problem.gene_edge_thickness_num] = new_gene_edge_thickness

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

    def make_inherit_genes_with_increasing_free_node(self, gene, solution):
        prior_node_num = self.prior_problem.node_num
        pro_node_num = self.problem.node_num

        # inherit edge_indices
        # どのエッジの間にノードを追加するかを選ぶ
        child_nodes_pos, child_edges_indices, child_gene_edges_thickness = self.prior_problem.get_graph_info_from_genes(gene)
        remove_indexes = [not self.check_equal_to_condition_edge(i) for i in child_edges_indices]
        candidate_indexes = np.where(remove_indexes)[0]
        selected_edge_index = np.random.choice(candidate_indexes)
        selected_edge_indices = child_edges_indices[selected_edge_index]
        selected_edge_indices = [2, 10]
        child_edges_indices = np.delete(child_edges_indices, selected_edge_index, 0)
        new_gene_edges_indices = np.concatenate([child_edges_indices, np.array([[selected_edge_indices[0], pro_node_num - 1], [selected_edge_indices[1], pro_node_num - 1]])])
        new_gene_edges_indices = revert_edge_indices_to_binary(new_gene_edges_indices, pro_node_num)
        solution.variables[-self.problem.gene_edge_indices_num:] = new_gene_edges_indices
        # inherit free node
        y_lower_bound = self.problem.distance_threshold
        selected_node1_pos = child_nodes_pos[selected_edge_indices[0]]
        selected_node2_pos = child_nodes_pos[selected_edge_indices[1]]
        vector = selected_node1_pos - selected_node2_pos
        # 追加するノードが下限のdistance_holdよりも下に行かないようにする．なお，これは固定ノード間など，distance_hold以下のノード同士を組んだエッジが選ばれないことを前提とする．
        if (selected_node1_pos[1] >= y_lower_bound) & (selected_node2_pos[1] >= y_lower_bound):
            r = np.random.rand(1)
        elif selected_node1_pos[1] >= y_lower_bound:
            limit = (y_lower_bound - selected_node2_pos[1]) / vector[1]
            r = np.random.uniform(low=limit, high=1.0)
        else:
            limit = (y_lower_bound - selected_node2_pos[1]) / vector[1]
            r = np.random.uniform(low=0.0, high=limit)
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
        return solution
