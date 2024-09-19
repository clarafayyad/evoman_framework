import random
import numpy as np
from deap import base, creator, tools

# Define problem type: maximize fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Define constants
IND_SIZE = 265  # Number of neural network parameters (weights + biases)
POP_SIZE = 100  # Population size
CXPB = 0.7  # Crossover probability
MUTPB = 0.2  # Mutation probability
TOUR_SIZE = 5  # Tournament size
ELITE_SIZE = 1  # Number of elites

# Toolbox setup
toolbox = base.Toolbox()

# Attribute generator: each weight is a float between -1 and 1
toolbox.register("attr_float", np.random.uniform, -1, 1, IND_SIZE)

# Structure initializers: define individuals and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function
def evaluate(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return f


toolbox.register("evaluate", evaluate)

# Selection: Tournament selection
toolbox.register("select_parents", tools.selTournament, tournsize=TOUR_SIZE)


# Crossover: Arithmetic crossover
def arithmetic_crossover(ind1, ind2):
    alpha = random.random()  # Random weight between [0, 1]
    for i in range(len(ind1)):
        ind1[i] = alpha * ind1[i] + (1 - alpha) * ind2[i]
        ind2[i] = alpha * ind2[i] + (1 - alpha) * ind1[i]
    return ind1, ind2


toolbox.register("mate", arithmetic_crossover)

# Mutation: Gaussian mutation
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)


# Survivor selection: Linear ranking with elitism
def select_survivors(population, k):
    return tools.selBest(population, k=ELITE_SIZE) + tools.selStochasticUniversalSampling(population, k - ELITE_SIZE)


toolbox.register("select_survivors", select_survivors)


# Main EA loop
def main():
    random.seed()

    # Create initial population
    pop = toolbox.population(n=POP_SIZE)

    # Run the evolutionary algorithm
    for gen in range(50):  # Set the number of generations
        # Select parents
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the fitness of the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        pop = toolbox.select_survivors(offspring, POP_SIZE)

        # Print best fitness of the generation
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Generation {gen}: Max fitness = {max(fits)}, Avg fitness = {np.mean(fits)}")

    return pop


if __name__ == "__main__":
    main()