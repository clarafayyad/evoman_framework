# imports
import global_env
import reporting
import stats
import operators
from evaluation import evaluate_population

class BasicEvolutionaryAlgorithm:
    def __init__(self, configs):
        self.configs = configs

    def execute_evolution(self, env):
        # Compute individual size
        individual_size = (env.get_num_sensors() + 1) * global_env.hidden_neurons + (global_env.hidden_neurons + 1) * 5

        # Initialize generation counter
        generation_number = 0

        # Initialize population
        population = operators.initialize_population(self.configs.population_size, individual_size,
                                                     global_env.lower_bound,
                                                     global_env.upper_bound)
        fitness_values = evaluate_population(env, population)

        # Save current population and fitness values
        env.update_solutions([population, fitness_values])

        # Compute and log stats
        best_individual_index, mean, std = stats.compute_stats(fitness_values)
        reporting.log_stats(global_env.experiment_name, generation_number, fitness_values[best_individual_index], mean,
                            std)

        # Evolution
        while generation_number < self.configs.total_generations:
            # Increment generation number
            generation_number += 1

            # Create offspring
            tournament_count = int(self.configs.population_size / 2)
            offspring = operators.arithmetic_uniform_crossover(
                population,
                fitness_values,
                tournament_count,
                self.configs.tournament_size,
                self.configs.crossover_weight,
                self.configs.crossover_rate,
            )

            # Mutate offspring
            for i in range(len(offspring)):
                # Apply gaussian mutation
                offspring[i] = operators.gaussian_mutation(offspring[i], rate=self.configs.mutation_rate,
                                                           sigma=self.configs.mutation_sigma)
                # Clamp the weights and biases within the initial range after applying variation operators
                offspring[i] = operators.clamp_within_bounds(offspring[i], global_env.lower_bound, global_env.upper_bound)

            # Evaluate offspring
            offspring_fitness = evaluate_population(env, offspring)

            # Survivor selection
            population, fitness_values = operators.linear_ranking_survivor_selection(
                population,
                fitness_values,
                offspring,
                offspring_fitness,
                s=self.configs.selection_pressure)

            # Compute and log stats
            best_individual_index, mean, std = stats.compute_stats(fitness_values)
            reporting.log_stats(global_env.experiment_name, generation_number, fitness_values[best_individual_index],
                                mean, std)

        # Save file with the best solution
        reporting.save_best_individual(global_env.experiment_name, population[best_individual_index],
                                       fitness_values[best_individual_index])

        # Update and save simulation state
        env.update_solutions([population, fitness_values])
        env.save_state()
