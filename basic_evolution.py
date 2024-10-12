# imports
import global_env
import reporting
import stats
import operators
from evaluation import evaluate_population


class BasicEvolutionaryAlgorithm:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.experiment = global_env.default_experiment_name
        self.run_number = 0

    def execute_evolution(self, env):
        # Compute individual size
        individual_size = (env.get_num_sensors() + 1) * global_env.hidden_neurons + (global_env.hidden_neurons + 1) * 5

        # Initialize generation counter
        generation_number = 0

        # Initialize population
        population = operators.initialize_population(
            self.hyperparams.population_size,
            individual_size,
            global_env.lower_bound,
            global_env.upper_bound)
        fitness_values = evaluate_population(env, generation_number, self.hyperparams.total_generations, population)

        # Save current population and fitness values
        env.update_solutions([population, fitness_values])

        # Compute and log stats
        # Compute basic fitness (non-dynamic) for logging purposes only (to have comparable results)
        basic_fitness_values = evaluate_population(
            env,
            generation_number,
            self.hyperparams.total_generations,
            population,
            force_basic_eval=True
        )
        best_individual_index, mean, std = stats.compute_stats(basic_fitness_values)
        reporting.log_stats(
            self.experiment,
            self.run_number,
            generation_number,
            basic_fitness_values[best_individual_index],
            mean,
            std
        )

        # Evolution
        while generation_number < self.hyperparams.total_generations:
            # Increment generation number
            generation_number += 1

            # Create offspring
            tournament_count = int(self.hyperparams.population_size / 2)
            offspring = operators.arithmetic_uniform_crossover(
                population,
                fitness_values,
                tournament_count,
                self.hyperparams.tournament_size,
                self.hyperparams.crossover_weight,
                self.hyperparams.crossover_rate,
            )

            # Mutate offspring
            for i in range(len(offspring)):
                # Apply gaussian mutation
                offspring[i] = operators.gaussian_mutation(
                    offspring[i], rate=self.hyperparams.mutation_rate,
                    sigma=self.hyperparams.mutation_sigma
                )
                # Clamp the weights and biases within the initial range after applying variation operators
                offspring[i] = operators.clamp_within_bounds(
                    offspring[i],
                    global_env.lower_bound,
                    global_env.upper_bound
                )

            # Evaluate offspring
            offspring_fitness = evaluate_population(
                env,
                generation_number,
                self.hyperparams.total_generations,
                offspring
            )

            # Survivor selection
            population, fitness_values = operators.linear_ranking_survivor_selection(
                population,
                fitness_values,
                offspring,
                offspring_fitness,
                s=self.hyperparams.selection_pressure)

            # For logging purposes only (to have comparable results)
            basic_fitness_values = evaluate_population(
                env,
                generation_number,
                self.hyperparams.total_generations,
                population,
                force_basic_eval=True
            )

            # Compute and log stats
            best_individual_index, mean, std = stats.compute_stats(basic_fitness_values)
            reporting.log_stats(
                self.experiment,
                self.run_number,
                generation_number,
                basic_fitness_values[best_individual_index],
                mean,
                std
            )

        # Save file with the best solution
        best_individual_index, _, _ = stats.compute_stats(fitness_values)
        reporting.save_best_individual(
            self.experiment,
            self.run_number,
            population[best_individual_index],
            fitness_values[best_individual_index],
        )

        # Update and save simulation state
        env.update_solutions([population, fitness_values])
        env.save_state()
