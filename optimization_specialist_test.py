# imports
import numpy as np
import sys

import operators


# loads file with the best solution for testing
def test_experiment(experiment_name, env):
    best_solution = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    operators.evaluate_population(env, [best_solution])
    sys.exit(0)
