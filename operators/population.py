import numpy as np


def initialize_population(population_size, variables_count, lower_bound, upper_bound):
    """
    Initializes a random population with individuals represented as arrays of floats.

    Args:
        population_size (int): Number of individuals in the population.
        variables_count (int): Number of variables (weights and biases) per individual.
        lower_bound (float): Minimum possible value for each variable.
        upper_bound (float): Maximum possible value for each variable.

    Returns:
        np.ndarray: A 2D array where each row is an individual in the population.
    """
    return np.random.uniform(lower_bound, upper_bound, (population_size, variables_count))
