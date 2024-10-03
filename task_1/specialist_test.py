# imports
import glob
import numpy as np
from reporting import log_test_results


# loads file with the best solution for testing
def test_experiment(env, apply_coevolution, enemy_number):
    folder_name = 'train_'
    if apply_coevolution:
        folder_name += 'ea2_e'
    else:
        folder_name += 'ea1_e'
    folder_name += str(enemy_number)

    if apply_coevolution:
        file_list = glob.glob(folder_name + '/main_network_results/best_ind*.txt')
    else:
        file_list = glob.glob(folder_name + '/best_ind*.txt')


    tests_per_individual = 5

    # Load each file and process it
    for file_name in file_list:
        best_solution = np.loadtxt(file_name)

        player_life_results = []
        enemy_life_results = []
        time_results = []

        for i in range(tests_per_individual):
            print('\n RUN #' + str(i+1) + ' FOR SOLUTION ' + file_name + '\n')
            env.update_parameter('speed', 'normal')
            _, player_life, enemy_life, time = env.play(pcont=best_solution)

            player_life_results.append(player_life)
            enemy_life_results.append(enemy_life)
            time_results.append(time)

        avg_player_life = np.mean(player_life_results)
        avg_enemy_life = np.mean(enemy_life_results)
        avg_time_result = np.mean(time_results)

        log_test_results(apply_coevolution, enemy_number, avg_player_life, avg_enemy_life, avg_time_result)

        player_life_results.clear()
        enemy_life_results.clear()
        time_results.clear()

