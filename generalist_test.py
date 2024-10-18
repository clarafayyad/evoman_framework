# imports
import glob
import numpy as np
import global_env
import reporting


# loads file with the best solution for testing
def test_experiment(env):
    env.enemies = [1, 2, 3, 4, 5, 6, 7, 8]  # test against all enemies
    env.update_parameter('speed', 'normal')

    ea_str = 'ea1_'
    if global_env.apply_dynamic_rewards:
        ea_str = 'ea2_'

    enemy_str = ','.join(map(str, global_env.enemies))

    train_folder = 'train_' + ea_str + enemy_str
    test_file = 'testing/test_' + ea_str + enemy_str + '.csv'
    file_list = glob.glob(train_folder + '/best_ind*.txt')

    tests_per_individual = 5

    # Load each file and process it
    for file_name in file_list:
        best_solution = np.loadtxt(file_name)

        player_life_results = []
        enemy_life_results = []
        time_results = []

        for i in range(tests_per_individual):
            print('\n RUN #' + str(i + 1) + ' FOR SOLUTION ' + file_name + ' AGAINST ENEMIES ' + ','.join(map(str, env.enemies)) + '\n')

            _, player_life, enemy_life, time = env.play(pcont=best_solution)

            player_life_results.append(player_life)
            enemy_life_results.append(enemy_life)
            time_results.append(time)

        avg_player_life = np.mean(player_life_results)
        avg_enemy_life = np.mean(enemy_life_results)
        avg_time_result = np.mean(time_results)

        reporting.log_test_results(test_file, avg_player_life, avg_enemy_life, avg_time_result)

        player_life_results.clear()
        enemy_life_results.clear()
        time_results.clear()
