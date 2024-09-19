import random
import numpy as np


def initialize_population(population_size, individual_size, lower_bound, upper_bound):
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


def fitness(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def tournament(population, k, env):
    """
       Perform tournament selection to choose the best individual from a random subset of the population.

       Parameters:
       population: A 2D array where each row is an individual in the population.
       k (int): The number of individuals to randomly sample from the population for the tournament.
       env: The game environment used to evaluate the fitness of each individual.

       Returns:
       numpy array: The individual with the highest fitness from the selected subset.
       """

    selected_individuals = random.sample(population.tolist(), k)

    # Find the best individual among the selected based on fitness
    best_individual = max(selected_individuals, key=lambda ind: fitness(env, ind))

    return best_individual


def parent_selection(population, k, env):
    p1 = tournament(population, k, env)
    p2 = tournament(population, k, env)
    return p1, p2


# Updates the population by selecting survivors.
def survivor_selection(original_population, offspring, fitness_values, env, s=1.5):
    # Add child to population and fitness
    population = np.concatenate((original_population, offspring))
    fitness_values = np.append(fitness_values, evaluate(env, offspring))

    # Order population and fitness from best to worst
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = population[sorted_indices]
    sorted_fitness = fitness_values[sorted_indices]

    # Selection based on linear rank
    mu = len(sorted_fitness)
    ranks = np.arange(mu)
    rank_probabilities = ((2 - s) / mu) + ((2 * ranks * (s - 1)) / (mu * (mu - 1)))
    rank_probabilities /= rank_probabilities.sum()

    selection = np.random.choice(mu, len(original_population), p=rank_probabilities, replace=False)

    # Elitism (preserves fittest individual)
    fittest_individual = np.argmax(fitness_values)
    if fittest_individual not in selection:
        selection[-1] = fittest_individual

    # Establish new population and fitness values
    new_population = sorted_population[selection]
    new_fitness_values = sorted_fitness[selection]

    return new_population, new_fitness_values


def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def evaluate(env, population):
    return np.array(list(map(lambda y: simulation(env, y), population)))


def crossover(env, population, tournament_count, tournament_size):
    """
        Apply random arithmetic crossover to create two children from two parents.

        Args:
        - parent1: A numpy array of the first parent's parameters.
        - parent2: A numpy array of the second parent's parameters.

        Returns:
        - offspring: A numpy array of the children's parameters.'
        """

    offspring = []

    for j in range(tournament_count):
        parent1, parent2 = parent_selection(population, tournament_size, env)

        # Initialize children
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Generate crossover
        for i in range(len(parent1)):
            alpha = np.random.rand()  # Random weight between 0 and 1
            child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            child2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]

        offspring.append(child1)
        offspring.append(child2)

    return np.array(offspring)


def mutate(child, rate=0.1, sigma=0.1):
    """
        Apply Gaussian mutation to neural network parameters.

        Args:
        - child: A numpy array of the neural network parameters.
        - rate: The probability of mutating each parameter.
        - sigma: Standard deviation of the Gaussian distribution for mutation.

        Returns:
        - mutated_child: A numpy array with mutated parameters.
        """

    mutated_child = np.copy(child)

    for i in range(len(mutated_child)):
        if np.random.rand() < rate:
            mutated_child[i] += mutated_child[i] + np.random.normal(0, sigma)

    return mutated_child
