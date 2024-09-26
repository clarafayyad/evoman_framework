import csv
import time
import numpy as np
import os

def save_generation(experiment_name, generation_number):
    generation_file = open(experiment_name + '/gen.txt', 'w')
    generation_file.write(str(generation_number))
    generation_file.close()


def retrieve_last_generation(experiment_name):
    generation_file = open(experiment_name + '/gen.txt', 'r')
    generation_number = int(generation_file.readline())
    generation_file.close()
    return generation_number


def log_stats(experiment_name, generation_number, max_fitness, mean, std):
    # Display relevant message
    print('\n GENERATION ' + str(generation_number)
          + ' ' + str(round(max_fitness, 6))
          + ' ' + str(round(mean, 6))
          + ' ' + str(round(std, 6)))

    # Log to file
    results_file_path = os.path.join(experiment_name, 'train_results.csv')
    with open(results_file_path, mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file)
        if generation_number == 0:
            results_writer.writerow(['gen', 'best', 'mean', 'std'])
        results_writer.writerow([generation_number, round(max_fitness, 6), round(mean, 6), round(std, 6)])

def log_sub_pop_stats(experiment_name, subpop_identifier, generation_number, max_fitness, mean, std):
    # Display relevant message
    print('\n SUBPOP ' + subpop_identifier
          + '\n ' + str(round(max_fitness, 6))
          + ' ' + str(round(mean, 6))
          + ' ' + str(round(std, 6)))

    # Log to file
    file_name = subpop_identifier + '_train_results.csv'
    results_file_path = os.path.join(experiment_name, file_name)
    with open(results_file_path, mode='a', newline='') as results_file:
        results_writer = csv.writer(results_file)
        if generation_number == 0:
            results_writer.writerow(['gen', 'best', 'mean', 'std'])
        results_writer.writerow([generation_number, round(max_fitness, 6), round(mean, 6), round(std, 6)])


def save_best_individual(experiment_name, best_individual, best_fitness):
    print('\n BEST FITNESS ' + str(best_fitness))
    np.savetxt(experiment_name + '/best.txt', best_individual)


def start_experiment(experiment_name, is_test=False):
    print('\nEXPERIMENT STARTED\n')
    if not is_test:
        create_logs_directory(experiment_name)
        clear_experiment_logs(experiment_name)
    ini_time = time.time()
    return ini_time


def end_experiment(execution_time):
    print('\nExecution time: ' + str(round(execution_time) / 60) + ' minutes')
    print('\nExecution time: ' + str(round(execution_time)) + ' seconds \n')
    print('\nEXPERIMENT COMPLETED\n')


def clean_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def clear_experiment_logs(experiment_name):
    clean_directory('./' + experiment_name)


def create_logs_directory(experiment_name):
    if not os.path.exists('./' + experiment_name):
        os.makedirs('./' + experiment_name)


def log_test_results(experiment_name, fitness, player_life, enemy_life, time):
    print(f"\nPlayer Life: {player_life}")
    print(f"Enemy Life: {enemy_life}")
    print(f"Time: {time}")
    if enemy_life > player_life :
        print("Result: Player Lost!")
    elif player_life > enemy_life:
        print("Result: Player Won!")
    else:
        print("Result: It's a Draw!")

    results_file = open(experiment_name + '/test_tesults.txt', 'a')
    results_file.write('\n\nfitness, player_life, enemy_life, time')
    results_file.write('\n' + str(round(fitness, 6))
                       + ' ' + str(round(player_life, 6))
                       + ' ' + str(round(enemy_life, 6))
                       + ' ' + str(round(time, 6)))
    results_file.close()
