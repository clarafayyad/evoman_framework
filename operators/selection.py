import random
import numpy as np


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

    selected_individuals = random.sample(population, k)

    # Find the best individual among the selected based on fitness
    best_individual = max(selected_individuals, key=lambda ind: fitness(env, ind['pcont']))

    return best_individual


def parent_selection(population, k, env):
    p1 = tournament(population, k, env)
    p2 = tournament(population, k, env)
    return p1, p2


# Updates the population by selecting survivors.
def survivor_selection(population, offspring, fitness_values, selection_pressure):
    pass


def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def evaluate(env, population):
    return np.array(list(map(lambda y: simulation(env, y), population)))
