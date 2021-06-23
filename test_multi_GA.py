from platypus import NSGAII, DTLZ2, ProcessPoolEvaluator
from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, PM, BitFlip
from GA.GA_class import Barfem_GA
import time


class DTLZ2_Slow(DTLZ2):

    def evaluate(self, solution):
        sum(range(1000000))
        super(DTLZ2_Slow, self).evaluate(solution)


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
    node_num = 85
    parent = (node_num * 2 + int(node_num * (node_num - 1) / 2) * 2)

    problem = Barfem_GA(node_num)

    start = time.time()
    # instantiate the optimization algorithm to run in parallel
    with ProcessPoolEvaluator(8) as evaluator:
        algorithm = NSGAII(problem, population_size=parent,
                           variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()), evaluator=evaluator)
        algorithm.step()
    print(algorithm.nfe)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
