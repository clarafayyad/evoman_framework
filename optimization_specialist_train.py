# imports
import optimization_specialist_test
from demo_controller import player_controller
from evoman.environment import Environment
import time
import reporting
import stats
import operators

# Define a set of constants
POPULATION_SIZE = 100
TOTAL_GENERATIONS = 10

# Set EA Operators Parameters
lower_bound = -1
upper_bound = 1
tournament_size = 5
mutation_rate = 0.2
mutation_sigma = 0.1
selection_pressure = 1.2

# NN configuration
hidden_neurons = 10

# Set experiment name
experiment = 'experiments'

# Initialize simulation
env = Environment(experiment_name=experiment,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Start experiment
is_test = False
ini_time = reporting.start_experiment(experiment, is_test)
if is_test:
    optimization_specialist_test.test_experiment(experiment, env)


# Compute individual size
individual_size = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5

# Log the environment state
env.state_to_log()

# Initialize generation counter
generation_number = 0

# Initialize population
population = operators.initialize_population(POPULATION_SIZE, individual_size, lower_bound, upper_bound)
fitness_values = operators.evaluate_population(env, population)

# Save current population and fitness values
env.update_solutions([population, fitness_values])

# Compute and log stats
best_individual_index, mean, std = stats.compute_stats(fitness_values)
reporting.log_stats(experiment, generation_number, fitness_values[best_individual_index], mean, std)


# Evolution
while generation_number < TOTAL_GENERATIONS:
    # Create offspring
    tournament_count = int(POPULATION_SIZE / 2)
    offspring = operators.random_arithmetic_crossover(population, fitness_values, tournament_count, tournament_size)

    # Mutate offspring
    for i in range(len(offspring)):
        offspring[i] = operators.gaussian_mutation(offspring[i], rate=mutation_rate, sigma=mutation_sigma)

    # Evaluate offspring
    offspring_fitness = operators.evaluate_population(env, offspring)

    # Survivor selection
    population, fitness_values = operators.linear_ranking_survivor_selection(
        population,
        fitness_values,
        offspring,
        offspring_fitness,
        s=selection_pressure)

    # Compute and log stats
    best_individual_index, mean, std = stats.compute_stats(fitness_values)
    reporting.log_stats(experiment, generation_number, fitness_values[best_individual_index], mean, std)

    # Save file with the best solution
    reporting.save_best_individual(experiment, population[best_individual_index])

    # Update and save simulation state
    env.update_solutions([population, fitness_values])
    env.save_state()

    generation_number += 1

# Print total execution time
reporting.log_execution_time(time.time() - ini_time)

# Log environment state
# env.state_to_log()
