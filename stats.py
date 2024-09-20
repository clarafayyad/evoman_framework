import numpy as np


def compute_stats(fitness_values):
    max_fitness_index = np.argmax(fitness_values)
    mean = np.mean(fitness_values)
    std = np.std(fitness_values)
    return max_fitness_index, mean, std
