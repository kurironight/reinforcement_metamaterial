import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numpy.core.numeric import _cross_dispatcher
from platypus import NSGAII, DTLZ2, ProcessPoolEvaluator
from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, UM, BitFlip, GeneticAlgorithm, PM
from GA.GA_class import *
from GA.algorithm import FixNode_NSGAII, FixNode_add_middle_point_NSGAII
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
"""
max_free_node_num = 18
free_nodes_num = np.arange(1, max_free_node_num + 1)
fix_nodes_num = np.ones(free_nodes_num.shape, dtype=np.int) * 8

"""
# 単体実行用
max_free_node_num = 6
free_nodes_num = np.arange(max_free_node_num, max_free_node_num + 1)
fix_nodes_num = np.ones(free_nodes_num.shape, dtype=np.int) * 8

experient_num = 5
GA_type = FixnodeForceDisp_GA
algorithm_type = FixNode_add_middle_point_NSGAII

if __name__ == "__main__":
    # define the problem definition
    save_dir = "GA/result/通常GA_n6_固定有"
    generation = 500
    save_interval = 5

    parent_mult_value = 10  # 遺伝子個数に対する親の個数の比率

    # PATH = os.path.join(save_dir, "parent_{}_gen_{}".format(parent, generation))
    # os.makedirs(PATH, exist_ok=False)
    for t in range(experient_num):
        PATH = os.path.join(save_dir, "{}".format(t))
        os.makedirs(PATH, exist_ok=True)

        start = time.time()
        # instantiate the optimization algorithm to run in parallel
        for index, (free_node_num, fix_node_num) in enumerate(zip(free_nodes_num, fix_nodes_num)):
            problem = GA_type(free_node_num, fix_node_num)
            parent = problem.nvars * parent_mult_value
            history = []
            GA_result_dir = os.path.join(PATH, "free_{}_fix_{}".format(free_node_num, fix_node_num))
            os.makedirs(GA_result_dir, exist_ok=True)
            with ProcessPoolEvaluator(8) as evaluator:
                if index == 0:  # 一つ目のGAでは遺伝子を引き継がない
                    problem = GA_type(free_node_num, fix_node_num)
                    algorithm = NSGAII(problem, population_size=parent, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()), evaluator=evaluator)
                else:  # 二回目以降のGAでは遺伝子を引き継ぐ
                    load_GA_result_dir = os.path.join(PATH, "free_{}_fix_{}".format(free_nodes_num[index - 1], fix_nodes_num[index - 1]))
                    load_gene_path = os.path.join(load_GA_result_dir, "parents.pk")
                    problem = GA_type(free_node_num, fix_node_num)
                    prior_problem = GA_type(free_nodes_num[index - 1], fix_nodes_num[index - 1])
                    algorithm = algorithm_type(problem, prior_problem, gene_path=load_gene_path, population_size=parent, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()), evaluator=evaluator)
                for i in tqdm(range(generation)):
                    algorithm.step()
                    efficiency_results = [s.objectives[0] for s in algorithm.result if s.feasible]  # 実行可能解しか抽出しない
                    if len(efficiency_results) != 0:
                        max_efficiency = max(efficiency_results)
                    else:
                        max_efficiency = problem.penalty_value
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
                    plt.savefig(os.path.join(GA_result_dir, "history.png"))
                    plt.close()

                    if (i + 1) % save_interval == 0:
                        save_log_dir = os.path.join(GA_result_dir, str(i + 1))
                        if not os.path.exists(save_log_dir):
                            os.makedirs(save_log_dir)
                        if len(efficiency_results) != 0:
                            max_index = efficiency_results.index(max_efficiency)
                            feasible_solution = [s for s in algorithm.result if s.feasible]
                            max_solution = feasible_solution[max_index]
                            vars = [problem.types[i].decode(max_solution.variables[i]) for i in range(problem.nvars)]
                            with open(os.path.join(save_log_dir, "parents.pk"), 'wb') as f:
                                pickle.dump(algorithm.result, f)
                            problem.calculate_efficiency(*problem.convert_var_to_arg(vars), np_save_dir=save_log_dir)
                        np.save(os.path.join(save_log_dir, "history.npy"), history)
                with open(os.path.join(GA_result_dir, "parents.pk"), 'wb') as f:
                    pickle.dump(algorithm.result, f)
            save_log_dir = os.path.join(GA_result_dir, "final")
            if not os.path.exists(save_log_dir):
                os.makedirs(save_log_dir)
            if len(efficiency_results) != 0:
                max_index = efficiency_results.index(max_efficiency)
                feasible_solution = [s for s in algorithm.result if s.feasible]
                max_solution = feasible_solution[max_index]
                vars = [problem.types[i].decode(max_solution.variables[i]) for i in range(problem.nvars)]
                with open(os.path.join(save_log_dir, "parents.pk"), 'wb') as f:
                    pickle.dump(algorithm.result, f)
                problem.calculate_efficiency(*problem.convert_var_to_arg(vars), np_save_dir=save_log_dir)
            np.save(os.path.join(save_log_dir, "history.npy"), history)
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
