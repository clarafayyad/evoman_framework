# imports
import numpy as np
import sys


# loads file with the best solution for testing
def test(experiment_name, env, evaluate):
    best_solution = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([best_solution])
    sys.exit(0)
