from numpy.core.numeric import _cross_dispatcher
from platypus import NSGAII, DTLZ2, ProcessPoolEvaluator
from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, UM, BitFlip, GeneticAlgorithm, PM
from GA.GA_class import *
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

"""
if __name__ == "__main__":
    # define the problem definition
    problem = DTLZ2_Slow()

    # instantiate the optimization algorithm to run in parallel
    with ProcessPoolEvaluator(4) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(10)

    # display the results
    for solution in algorithm.result:
        print(solution.objectives)
"""

if __name__ == "__main__":
    # define the problem definition
    save_dir = "GA/result"
    node_num = 2 + 2 + 2
    #node_num = 40
    parent = (node_num * 2 + int(node_num * (node_num - 1) / 2) * 2)
    #parent = 12
    generation = 5000
    save_interval = 100

    PATH = os.path.join(save_dir, "parent_{}_gen_{}".format(parent, generation))
    os.makedirs(PATH, exist_ok=False)
    #PATH = os.path.join(save_dir, "test")
    #os.makedirs(PATH, exist_ok=True)

    problem = ConstraintIncrementalNodeIncrease_GA(2, 2)

    history = []

    start = time.time()
    # instantiate the optimization algorithm to run in parallel
    with ProcessPoolEvaluator(8) as evaluator:
        #algorithm = NSGAII(problem, population_size=parent, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()), evaluator=evaluator)
        algorithm = GeneticAlgorithm(problem, population_size=parent, offspring_size=parent,
                                     variator=CompoundOperator(SBX(), HUX(), UM(), BitFlip()), evaluator=evaluator)
        for i in tqdm(range(generation)):
            algorithm.step()
            """
            nondominated_solutions = nondominated(algorithm.result)
            efficiency_results = [s.objectives[0] for s in nondominated_solutions]
            max_efficiency = max(efficiency_results)
            """
            max_efficiency = algorithm.fittest.objectives[0]
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

            if algorithm.nfe / parent % save_interval == 0:
                save_dir = os.path.join(PATH, str(i + 1))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                """
                max_index = efficiency_results.index(max_efficiency)
                max_solution = nondominated_solutions[max_index]
                """
                max_solution = algorithm.fittest

                vars = [problem.types[i].decode(max_solution.variables[i]) for i in range(problem.nvars)]
                problem.calculate_efficiency(*problem.convert_var_to_arg(vars), np_save_dir=save_dir)
                np.save(os.path.join(save_dir, "history.npy"), history)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
