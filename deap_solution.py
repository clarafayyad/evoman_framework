# imports
import numpy as np

from evoman.environment import Environment
import time
import os
import random
from deap_toolbox import toolbox

# Define all consts
TOTAL_GENERATIONS = 50
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.2


# Initialize simulation
env = Environment()

# Display env state
env.state_to_log()

# Start timer
ini = time.time()

# Initialize population loading old solutions or generating new ones
pop = np.empty(POPULATION_SIZE)
experiment_name = ''
if not os.path.exists(experiment_name + '/evoman_solstate'):
    print('\nNEW EVOLUTION\n')
    # generate new population
    pop = toolbox.population(n=POPULATION_SIZE)
else:
    print('\nCONTINUING EVOLUTION\n')
    # load from env

# save results in a txt file and output/print some relevant messages
# code here

random.seed()

# evolution
generation_count = 0
while generation_count < TOTAL_GENERATIONS:
    # parent selection
    parents = toolbox.select_parents(pop, POPULATION_SIZE)
    if len(parents) < 2:
        print('\nError: less than 2 parents selected\n')
        break

    # create offspring
    offspring = toolbox.mate(parents[0], parents[1])
    for mutant in offspring:
        if random.random() < MUTATION_PROBABILITY:
            toolbox.mutate(mutant)

    # evaluate offspring
    fitness_values = map(toolbox.evaluate, offspring)

    # survivor selection
    survivors = toolbox.select_survivors(offspring, POPULATION_SIZE)

    # save results in the txt file and output relevant msgs
    # save file with the best solution
    # save simulation state

    generation_count += 1

# print total execution time for experiment

# save control (simulation has ended) file for bash loop file

# check environment state
