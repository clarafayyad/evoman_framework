import random
import numpy as np


def fitness(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def tournament(population, k, env):
    selected_individuals = random.sample(population, k)
    # Find the best individual among the selected based on fitness
    best_individual = max(selected_individuals, key=lambda ind: fitness(env, ind['pcont']))
    
    return best_individual

# Selects 2 parents from the current population, choose k=3 or k=5
def parent_selection(population, k, env):
    p1 = tournament(population, k, env)
    p2 = tournament(population, k, env)
    return p1, p2

# Updates the population by selecting survivors.
def survivor_selection(population, offspring, fitness_values, env, s=1.5):
    #Add child to population and fitness
    population = np.vstack((population, offspring))
    fitness_values = np.append(fitness_values, fitness(env, offspring))

    #Order population and fitness from best to worst
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = population[sorted_indices]
    sorted_fitness = fitness_values[sorted_indices]

    #Selection based on linear rank
    mu = len(sorted_fitness)
    ranks = np.arange(mu)
    rank_probabilities = ((2 - s) / mu) + ((2 * ranks * (s - 1)) / (mu * (mu - 1)))

    rank_probabilities /= rank_probabilities.sum()

    selection = np.random.choice(mu, mu, p=rank_probabilities, replace=False)

    #Elitism (preserves fittest individual)
    fittest_individual = np.argmax(fitness_values)
    if fittest_individual not in selection:
        selection[-1] = fittest_individual

    #Establish new population and fitness values
    new_population = sorted_population[selection]
    new_fitness_values = sorted_fitness[selection]

    return new_population, new_fitness_values