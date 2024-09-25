# imports
import numpy as np
import reporting
import stats
import operators

# Define a set of constants
POPULATION_SIZE = 100
TOTAL_GENERATIONS = 30

# Set EA Operators Parameters
lower_bound = -1
upper_bound = 1
tournament_size = 5
mutation_rate = 0.3
mutation_sigma = 0.3
selection_pressure = 1
crossover_weight = 0.8
crossover_rate = 0.5
sigma_share = 0.8  # niche radius


def basic_evolution(experiment, env, hidden_neurons):
    # Compute individual size
    individual_size = (env.get_num_sensors() + 1) * hidden_neurons + (hidden_neurons + 1) * 5

    # Log the environment state
    # env.state_to_log()

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
        # Increment generation number
        generation_number += 1

        # Create offspring
        tournament_count = int(POPULATION_SIZE / 2)
        offspring = operators.arithmetic_uniform_crossover(
            population,
            fitness_values,
            tournament_count,
            tournament_size,
            crossover_weight,
            crossover_rate,
        )

        # Mutate offspring
        for i in range(len(offspring)):
            # Apply gaussian mutation
            offspring[i] = operators.gaussian_mutation(offspring[i], rate=mutation_rate, sigma=mutation_sigma)
            # Clamp the weights and biases within the initial range after applying variation operators
            operators.clamp_within_bounds(offspring[i], lower_bound, upper_bound)

        # Evaluate offspring
        offspring_fitness = operators.evaluate_population(env, offspring)

        # Apply fitness sharing
        # fitness_values, offspring_fitness = operators.fitness_sharing(
        #     population,
        #     fitness_values,
        #     offspring,
        #     offspring_fitness,
        #     sigma_share
        # )

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

