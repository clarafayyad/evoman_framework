# imports
import numpy as np
from reporting import log_test_results


# loads file with the best solution for testing
def test_experiment(experiment_name, env):
    best_solution = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    fitness, player_life, enemy_life, time = env.play(pcont=best_solution)
    log_test_results(experiment_name, fitness, player_life, enemy_life, time)
