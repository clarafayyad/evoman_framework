import numpy as np

import global_env
import operators
import reporting
import stats
from multiprocessing import Pool
from evoman.environment import Environment
from evaluation import evaluate_individual, evaluate_objectives_and_fitness

# Define subnetworks identifiers
FEATURES_POP = 'input_to_hidden'
WALK_LEFT_POP = 'walk_left'
WALK_RIGHT_POP = 'walk_right'
JUMP_POP = 'jump'
SHOOT_POP = 'shoot'
RELEASE_POP = 'release'

class Subpopulation:
    def __init__(self, identifier, individuals, configs):
        self.identifier = identifier
        self.individuals = individuals
        self.fitness = np.zeros(len(individuals))
        # Select a random individual as the initial best individual
        self.best_individual = self.individuals[np.random.randint(len(individuals))]
        self.objectives = np.array([np.array([0, 0, 0]) for _ in range(len(individuals))])
        self.configs = configs

    def evaluate(self, env, best_subnetworks):
        for i, individual in enumerate(self.individuals):
            # Combine current individual with the best individuals from the other subnetworks
            full_network = combine_subnetworks(self.identifier, individual, best_subnetworks)
            # Evaluate full network
            self.objectives[i], self.fitness[i] = evaluate_objectives_and_fitness(env, full_network)

        # Find the best individual based on pareto front
        self.best_individual = self.individuals[operators.select_best_pareto_individual(self.objectives)]

    def evolve(self, env, generation_number, best_subnetworks):
        # Evaluate current subnetwork/subpopulation
        self.evaluate(env, best_subnetworks)

        # Create Offspring
        tournament_count = int(len(self.individuals) / 2)
        offspring = operators.arithmetic_uniform_crossover(
            self.individuals,
            self.fitness,
            tournament_count,
            self.configs.tournament_size,
            self.configs.crossover_weight,
            self.configs.crossover_rate,
        )

        # Mutate offspring
        for i in range(len(offspring)):
            # Apply gaussian mutation
            offspring[i] = operators.gaussian_mutation(offspring[i], rate=self.configs.mutation_rate, sigma=self.configs.mutation_sigma)
            # Clamp the weights and biases within the initial range after applying variation operators
            offspring[i] = operators.clamp_within_bounds(offspring[i], global_env.lower_bound, global_env.upper_bound)

        # Evaluate offspring
        offspring_sub_pop = Subpopulation(self.identifier, offspring, self.configs)
        offspring_sub_pop.evaluate(env, best_subnetworks)

        combined_individuals = np.concatenate((self.individuals, offspring_sub_pop.individuals), axis=0)
        combined_fitness = np.concatenate((self.fitness, offspring_sub_pop.fitness), axis=0)
        combined_objectives = np.concatenate((self.objectives, offspring_sub_pop.objectives), axis=0)

        # Pareto-based Survivor selection
        selected_indices = operators.pareto_based_survivor_selection(combined_objectives, self.configs.population_size)
        self.individuals = combined_individuals[selected_indices]
        self.objectives = combined_objectives[selected_indices]
        self.fitness = combined_fitness[selected_indices]
        self.best_individual = self.individuals[operators.select_best_pareto_individual(self.objectives)]

        # Compute and log stats
        best_individual_index, mean, std = stats.compute_stats(self.fitness)
        reporting.log_sub_pop_stats(global_env.experiment_name, self.identifier, generation_number, self.fitness[best_individual_index], mean, std)


def combine_subnetworks(current_pop_id, current_individual, best_subnetworks):
    network_order = [FEATURES_POP, WALK_LEFT_POP, WALK_RIGHT_POP, JUMP_POP, SHOOT_POP, RELEASE_POP]

    best_subnetworks[current_pop_id] = current_individual

    combined_network = []
    for key in network_order:
        combined_network.append(best_subnetworks[key])

    return np.hstack(combined_network)


def initialize_random_sub_population(identifier, configs, individual_size):
    return Subpopulation(identifier, operators.initialize_population(
        configs.population_size,
        individual_size,
        global_env.lower_bound,
        global_env.upper_bound
    ), configs)


def evolve_subpop(subpop, generation, best_subnetworks):
    env = Environment(
        experiment_name=global_env.experiment_name,
        enemies=global_env.enemies,
        multiplemode=global_env.multiple_mode,
        playermode=global_env.player_mode,
        player_controller=global_env.player_controller,
        enemymode=global_env.enemy_mode,
        level=global_env.level,
        speed=global_env.speed,
        randomini=global_env.random_ini,
        visuals=global_env.visuals
    )
    subpop.evolve(env, generation, best_subnetworks)
    return subpop

class CoevolutionaryMultiObjAlgorithm:
    def __init__(self, configs):
        self.configs = configs

    def cooperative_coevolution(self, env):
        input_to_hidden_size = (env.get_num_sensors() + 1) * global_env.hidden_neurons
        hidden_to_output_size = global_env.hidden_neurons + 1

        # Initialize subpopulations
        features_pop = initialize_random_sub_population(FEATURES_POP, self.configs, input_to_hidden_size)
        walk_left_pop = initialize_random_sub_population(WALK_LEFT_POP, self.configs, hidden_to_output_size)
        walk_right_pop = initialize_random_sub_population(WALK_RIGHT_POP, self.configs, hidden_to_output_size)
        jump_pop = initialize_random_sub_population(JUMP_POP, self.configs, hidden_to_output_size)
        shoot_pop = initialize_random_sub_population(SHOOT_POP, self.configs, hidden_to_output_size)
        release_pop = initialize_random_sub_population(RELEASE_POP, self.configs, hidden_to_output_size)

        subpopulations = [features_pop, walk_left_pop, walk_right_pop, jump_pop, shoot_pop, release_pop]
        subpopulations_len = len(subpopulations)

        best_individual_found = None
        best_fitness_found = 0

        # Co-evolution
        for generation in range(self.configs.total_generations + 1):
            print('\nGENERATION ', generation)

            # Evolve each subpopulation
            for i in range(subpopulations_len):
                # Get the best individuals from the other subpopulations
                other_best_subnetworks = {}
                for j in range(subpopulations_len):
                    if j != i:
                        other_best_subnetworks[subpopulations[j].identifier] = subpopulations[j].best_individual

                # Evolve current subpopulation by evaluating it with the best individuals from the other subpopulations
                subpopulations[i].evolve(env, generation, other_best_subnetworks)

            # Create best network out of subpopulations
            current_best_network = np.hstack([subpop.best_individual for subpop in subpopulations])
            current_best_fitness = evaluate_individual(env, current_best_network)
            reporting.log_stats(global_env.experiment_name, generation, current_best_fitness, 0, 0)

            if current_best_fitness > best_fitness_found:
                best_individual_found = current_best_network
                best_fitness_found = current_best_fitness

        reporting.save_best_individual(global_env.experiment_name, best_individual_found, best_fitness_found)





