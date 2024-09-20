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
    results_file = open(experiment_name + '/results.txt', 'a')
    results_file.write('\n\ngen best mean std')
    results_file.write('\n' + str(generation_number)
                       + ' ' + str(round(max_fitness, 6))
                       + ' ' + str(round(mean, 6))
                       + ' ' + str(round(std, 6)))
    results_file.close()


def save_best_individual(experiment_name, best_individual):
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

