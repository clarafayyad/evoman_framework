# imports
import numpy as np
from reporting import log_test_results


# loads file with the best solution for testing
def test_experiment(env, apply_coevolution, enemy_number):
    file_name = 'best_ind_'
    if apply_coevolution:
        file_name += 'ea2_e'
    else:
        file_name += 'ea1_e'
    file_name += str(enemy_number)
    file_name += '.txt'

    best_solution = np.loadtxt('best_inds_for_testing/' + file_name)
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    fitness, player_life, enemy_life, time = env.play(pcont=best_solution)
    log_test_results(apply_coevolution, enemy_number, player_life, enemy_life, time)
