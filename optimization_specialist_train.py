# imports
from evoman.environment import Environment
import time
import os
import numpy as np
from operators.selection import evaluate, parent_selection, survivor_selection
from operators.variation import crossover, mutate

# Define a set of constants
POPULATION_SIZE = 100
INDIVIDUAL_SIZE = 265
TOTAL_GENERATIONS = 50

# Set Parameters
lower_bound = -1
upper_bound = 1
k = 5  # tournament_size
mutation_rate = 0.2
mutation_sigma = 0.1
selection_pressure = 1.5

# initialize simulation
env = Environment()

# check environment state
env.state_to_log()

# set time marker
ini = time.time()

# Initialize generation counter
generation_count = 0

# Initialize population loading old solutions or generating new ones
experiment_name = ''
if not os.path.exists(experiment_name + '/evoman_solstate'):
    print('\nNEW EVOLUTION\n')
    # Generate new population here
    population = np.random.uniform(lower_bound, upper_bound, (POPULATION_SIZE, INDIVIDUAL_SIZE))
    population_fitness = evaluate(env, population)
    env.update_solutions([population, population_fitness])
else:
    print('\nCONTINUING EVOLUTION\n')
    # Load generation from env
    env.load_state()
    population = env.solutions[0]
    population_fitness = env.solutions[1]

    # Last generation count
    generation_file = open(experiment_name + '/gen.txt', 'r')
    generation_count = int(generation_file.readline())
    generation_file.close()

best_individual_index = np.argmax(population_fitness)
mean = np.mean(population_fitness)
std = np.std(population_fitness)

# Save results in a txt file
results_file = open(experiment_name + '/results.txt', 'a')
results_file.write('\n\ngen best mean std')
results_file.write('\n' + str(generation_count)
                   + ' ' + str(round(population_fitness[best_individual_index], 6))
                   + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
results_file.close()

# Display current generation stats
print('\n GENERATION ' + str(generation_count)
      + ' ' + str(round(population_fitness[best_individual_index], 6))
      + ' ' + str(round(mean, 6))
      + ' ' + str(round(std, 6)))

# Evolution
while generation_count < TOTAL_GENERATIONS:
    # Select parents
    parent1, parent2 = parent_selection(population, k, env)

    # Create offspring
    child1, child2 = crossover(parent1, parent2)

    # Mutate offspring
    child1 = mutate(child1, rate=mutation_rate, sigma=mutation_sigma)
    child2 = mutate(child2, rate=mutation_rate, sigma=mutation_sigma)

    # Survivor selection
    population, population_fitness = survivor_selection(population, [child1, child2], population_fitness,
                                                        selection_pressure)

    best_individual_index = np.argmax(population_fitness)
    mean = np.mean(population_fitness)
    std = np.std(population_fitness)

    # Save results in the txt file
    results_file = open(experiment_name + '/results.txt', 'a')
    results_file.write('\n' + str(generation_count)
                       + ' ' + str(round(population_fitness[best_individual_index], 6))
                       + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    results_file.close()

    print('\n GENERATION ' + str(generation_count)
          + ' ' + str(round(population_fitness[best_individual_index], 6))
          + ' ' + str(round(mean, 6))
          + ' ' + str(round(std, 6)))

    # Save generation number
    generation_file = open(experiment_name + '/gen.txt', 'w')
    generation_file.write(str(generation_count))
    generation_file.close()

    # Save file with the best solution
    np.savetxt(experiment_name + '/best.txt', population[best_individual_index])

    # Save simulation state
    env.update_solutions([population, population_fitness])
    env.save_state()

    generation_count += 1

# Print total execution time
done = time.time()
print('\nExecution time: ' + str(round((done - ini) / 60)) + ' minutes \n')
print('\nExecution time: ' + str(round((done - ini))) + ' seconds \n')

# Simulation has ended
file = open(experiment_name+'/neuroended', 'w')
file.close()

# Log environment state
env.state_to_log()
