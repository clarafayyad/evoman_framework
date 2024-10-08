import random
import numpy as np


def initialize_population(population_size, individual_size, lower_bound=-1, upper_bound=1):
    """
    Initializes a random population with individuals represented as arrays of floats.

    Args:
        population_size (int): Number of individuals in the population.
        individual_size (int): Number of variables (weights and biases) per individual.
        lower_bound (float): Minimum possible value for each variable.
        upper_bound (float): Maximum possible value for each variable.

    Returns:
        np.ndarray: A 2D array where each row is an individual in the population.
    """
    return np.random.uniform(lower_bound, upper_bound, (population_size, individual_size))


def tournament(population, fitness_values, tournament_size):
    """
    Perform tournament selection to choose the best individual from a random subset of the population.
    :param population: A 2D array where each row is an individual in the population.
    :param fitness_values: Precomputed fitness values for each individual.
    :param tournament_size: The number of individuals to randomly sample from the population for the tournament.
    :return: numpy array: The individual with the highest fitness from the selected subset.
    """

    # Randomly sample k indices from the population
    selected_indices = random.sample(range(len(population)), tournament_size)

    # Find the index of the individual with the highest fitness in the selected subset
    best_index = max(selected_indices, key=lambda idx: fitness_values[idx])

    # Return the individual with the highest fitness
    return population[best_index]


def tournament_parent_selection(population, fitness_values, tournament_size):
    parent1 = tournament(population, fitness_values, tournament_size)
    parent2 = tournament(population, fitness_values, tournament_size)
    return parent1, parent2


def linear_ranking_survivor_selection(original_population, pop_fitness, offspring, offspring_fitness, s=1.5):
    """
    Perform survivor selection using linear ranking with elitism.
    :param original_population: A 2D array where each row is an individual in the original population.
    :param pop_fitness: A 1D array of fitness values for the original population.
    :param offspring: A 2D array where each row is an individual from the offspring.
    :param offspring_fitness: A 1D array of fitness values for the offspring.
    :param s: Selective pressure parameter (1 < s <= 2), used for linear ranking.
    :return: 2 numpy arrays: The new population after survivor selection, and the corresponding fitness values.
    """

    # Concatenate original population with offspring
    combined_population = np.concatenate((original_population, offspring), axis=0)
    combined_fitness = np.concatenate((pop_fitness, offspring_fitness), axis=0)

    # Find the elite individual and its index
    elite_index = np.argmax(combined_fitness)
    elite_individual = combined_population[elite_index]
    elite_fitness = combined_fitness[elite_index]

    # Remove the elite individual from the combined population and fitness
    combined_population = np.delete(combined_population, elite_index, axis=0)
    combined_fitness = np.delete(combined_fitness, elite_index)

    # Sort remaining individuals by fitness to prepare for ranking
    sorted_indices = np.argsort(-combined_fitness)
    sorted_population = combined_population[sorted_indices]
    sorted_fitness = combined_fitness[sorted_indices]

    # Linear ranking probabilities
    n = len(sorted_population)
    ranks = np.arange(1, n + 1)  # Rank from 1 to n
    probabilities = (2 - s) / n + (2 * ranks * (s - 1)) / (n * (n - 1))
    probabilities /= np.sum(probabilities)  # Normalize the probabilities to sum to 1

    # Add the elite individual to the new population
    new_population = [elite_individual]
    new_fitness_values = [elite_fitness]

    # Select the remaining individuals based on linear ranking probabilities
    selection_size = len(original_population) - 1
    selected_indices = np.random.choice(np.arange(n), size=selection_size, replace=False, p=probabilities)

    # Add the selected individuals and their fitness values to the new population
    new_population.extend(sorted_population[selected_indices])
    new_fitness_values.extend(sorted_fitness[selected_indices])

    return np.array(new_population), np.array(new_fitness_values)


def clamp_within_bounds(values, lower_bound, upper_bound):
    """
    Clamp the values within a certain range.
    :param values: Numpy array of floats representing the values.
    :param lower_bound: The minimum allowable value.
    :param upper_bound: The maximum allowable value.
    :return: Numpy array of floats representing the clamped values.
    """
    for i in range(len(values)):
        if values[i] > upper_bound:
            values[i] = upper_bound
        if values[i] < lower_bound:
            values[i] = lower_bound
    return values


def arithmetic_uniform_crossover(population, fitness_values, tournament_count, tournament_size, alpha, rate):
    """
    Apply random arithmetic crossover to create offspring.
    :param population: Numpy array representing the population.
    :param fitness_values: Numpy array representing the fitness values.
    :param tournament_count: Number of rounds to crossover.
    :param tournament_size: Number of individuals to randomly sample from the population for each tournament.
    :param alpha: The weight used for the arithmetic crossover.
    :param rate: The probability to swap genes between children, used for uniform crossover.
    :return: Numpy array representing the offspring.
    """

    offspring = []

    for j in range(tournament_count):
        parent1, parent2 = tournament_parent_selection(population, fitness_values, tournament_size)

        # Initialize children
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Generate crossover
        for i in range(len(parent1)):
            child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            child2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]

            # Apply uniform crossover
            if np.random.rand() < rate:
                child1[i], child2[i] = child2[i], child1[i]

        offspring.append(child1)
        offspring.append(child2)

    return np.array(offspring)


def gaussian_mutation(child, rate=0.1, sigma=0.1):
    """
    Apply gaussian mutation.
    :param child: A numpy array representing the child individual.
    :param rate: The probability that the child individual will be mutated.
    :param sigma: Standard deviation of the gaussian distribution for mutation.
    :return: A numpy array representing the mutated child individual.
    """

    mutated_child = np.copy(child)

    for i in range(len(mutated_child)):
        if np.random.rand() < rate:
            mutated_child[i] += mutated_child[i] + np.random.normal(0, sigma)

    return mutated_child


def fitness_sharing(population, pop_fitness, offspring, offspring_fitness, sigma_share):
    """
    Applies fitness sharing to the combined population of parents and offspring.
    :param population: 2D Numpy array representing the current population.
    :param pop_fitness: Numpy array of shape representing the fitness values of the population.
    :param offspring: 2D Numpy array representing the offspring.
    :param offspring_fitness: Numpy array representing the fitness values of the offspring.
    :param sigma_share: Float value representing the niche radius.
    :return:
        - shared_fitness_population: Numpy array representing the shared fitness values of the population.
        - shared_fitness_offspring: Numpy array representing the shared fitness values of the offspring.
    """

    # Combine parents and offspring into a single population
    combined_population = np.vstack((population, offspring))
    combined_fitness = np.concatenate((pop_fitness, offspring_fitness))

    shared_fitness = np.copy(combined_fitness)
    n = len(combined_population)

    for i in range(n):
        sharing_sum = 0.0

        for j in range(n):
            if i != j:
                distance = np.linalg.norm(combined_population[i] - combined_population[j])
                sharing_value = sharing_function(distance, sigma_share)
                sharing_sum += sharing_value

        if sharing_sum > 0:
            shared_fitness[i] = combined_fitness[i] / sharing_sum

    shared_fitness_population = shared_fitness[:len(population)]
    shared_fitness_offspring = shared_fitness[len(population):]

    return shared_fitness_population, shared_fitness_offspring


def sharing_function(distance, sigma_share):
    """
    Sharing function based on distance between individuals.
    :param distance: The distance between two individuals.
    :param sigma_share: The niche radius.
    :return: Sharing value (between 0 and 1).
    """
    if distance < sigma_share:
        return 1 - (distance / sigma_share)
    else:
        return 0.0


def pareto_based_survivor_selection(individuals, objectives, num_survivors):
    pareto_front = pareto_sort(objectives)

    # Randomly select n individuals from the first Pareto front
    selected_indices = np.random.choice(pareto_front, size=min(num_survivors, len(pareto_front)), replace=False)

    # Collect the selected individuals and their objectives
    selected_individuals = np.array(individuals[selected_indices])
    selected_objectives = np.array(objectives[selected_indices])

    return selected_individuals, selected_objectives

def pareto_sort(population_objectives):
    num_individuals = len(population_objectives)
    pareto_fronts = []
    domination_count = np.zeros(num_individuals)
    dominated_solutions = [[] for _ in range(num_individuals)]

    for i in range(num_individuals):
        for j in range(num_individuals):
            if i != j:
                if dominates(population_objectives[i], population_objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates(population_objectives[j], population_objectives[i]):
                    domination_count[i] += 1

        if domination_count[i] == 0:
            pareto_fronts.append(i)

    return pareto_fronts

def dominates(obj1, obj2):
    return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 > o2 for o1, o2 in zip(obj1, obj2))

def select_best_pareto_individual(fitness):
    ideal_point = np.max(fitness, axis=0)
    distances = np.linalg.norm(fitness - ideal_point, axis=1)
    best_index = np.argmin(distances)
    return best_index
