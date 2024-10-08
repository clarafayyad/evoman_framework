import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from multiprocessing import Pool

from evoman.environment import Environment
from evaluation import reward_objectives, evaluate_individual, select_best_pareto_individual
import global_env
import operators
import reporting
import stats


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
        self.fitness = np.zeros(len(individuals)) # This is only used for stats, not for evaluation or selection
        self.objectives = np.array([])
        self.best_individual = self.individuals[np.random.randint(len(individuals))]
        self.mean = 0
        self.std = 0
        self.configs = configs

    def evaluate(self, env, best_subnetworks):
        for i, individual in enumerate(self.individuals):
            # Combine current individual with the best individuals from the other subnetworks
            full_network = combine_subnetworks(self.identifier, individual, best_subnetworks)
            # Evaluate full network
            self.fitness[i] = evaluate_individual(env, full_network)

        # Find the best individual based on fitness
        self.best_individual = self.individuals[np.argmax(self.fitness)]

    def evolve(self, env, generation_number, best_subnetworks):
        problem = NSGA2Problem(self.individuals, self.configs, env, best_subnetworks)
        algorithm = NSGA2(pop_size=len(self.individuals))

        res = minimize(problem, algorithm, termination=('n_gen', self.configs.total_generations), seed=1)

        self.individuals = res.X
        self.objectives = res.F
        self.best_individual = select_best_pareto_individual(self.objectives)

        # Compute and log stats
        self.evaluate(env, best_subnetworks)
        best_individual_index, self.mean, self.std = stats.compute_stats(self.fitness)
        reporting.log_sub_pop_stats(global_env.experiment_name, self.identifier, generation_number,
                                    self.fitness[best_individual_index], self.mean, self.std)


class NSGA2Problem(Problem):
    def __init__(self, individuals, identifier, env, best_subnetworks):
        n_var = len(individuals[0])
        super().__init__(n_var=n_var, n_obj=4, xl=global_env.lower_bound, xu=global_env.upper_bound)

        self.individuals = individuals
        self.identifier = identifier
        self.env = env
        self.best_subnetworks = best_subnetworks

    def _evaluate(self, x, out, *args, **kwargs):
        fitness_objectives = []
        for individual in x:
            full_network = combine_subnetworks(self.identifier, individual, self.best_subnetworks)
            objectives = reward_objectives(self.env, full_network)
            fitness_objectives.append(objectives)
        out["F"] = np.array(fitness_objectives)

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

        best_individual_found = None
        best_fitness_found = 0

        # Co-evolution
        for generation in range(self.configs.total_generations):
            print('\nGENERATION ', generation)

            # Prepare a dictionary of the best individuals for each subpopulation
            best_subnetworks = {subpop.identifier: subpop.best_individual for subpop in subpopulations}

            # Evolve current subpopulation by evaluating it with the best individuals from the other subpopulations
            args_list = [(subpop, generation, best_subnetworks) for subpop in subpopulations]
            with Pool(processes=len(subpopulations)) as pool:
                subpopulations = pool.starmap(evolve_subpop, args_list)
                pool.close()
                pool.join()

            current_best_network = np.hstack([subpop.best_individual for subpop in subpopulations])
            current_best_fitness = evaluate_individual(env, current_best_network)
            current_avg_mean = np.mean([subpop.mean for subpop in subpopulations])
            current_avg_std = np.mean([subpop.std for subpop in subpopulations])
            reporting.log_stats(global_env.experiment_name, generation, current_best_fitness, current_avg_mean, current_avg_std)

            if current_best_fitness > best_fitness_found:
                best_individual_found = current_best_network
                best_fitness_found = current_best_fitness

        reporting.save_best_individual(global_env.experiment_name, best_individual_found, best_fitness_found)





