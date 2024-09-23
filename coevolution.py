import numpy as np
import operators
import parameters
from parameters import lower_bound, upper_bound

FEATURES_POP = 'input_to_hidden'
WALK_LEFT_POP = 'walk left'
WALK_RIGHT_POP = 'walk right'
JUMP_POP = 'jump'
SHOOT_POP = 'shoot'
RELEASE_POP = 'release'


class Subpopulation:
    def __init__(self, name, num_params):
        # Initialize with random individuals
        self.individuals = operators.initialize_population(
            parameters.POPULATION_SIZE,
            num_params,
            lower_bound,
            upper_bound
        )
        # Store fitness for each individual
        self.fitness = np.zeros(parameters.POPULATION_SIZE)
        # Select a random individual as the initial best individual
        self.best_individual = self.individuals[np.random.randint(parameters.POPULATION_SIZE)]
        self.name = name

    def evaluate(self, env, other_best_subnetworks):
        """
        Evaluate each individual in this subpopulation.
        other_best_subnetworks: list of the best individuals from other subpopulations.
        evaluate_fn: function to evaluate the combined full network in the game.
        """
        for i, individual in enumerate(self.individuals):
            # Combine current individual with the best individuals from the other subnetworks
            full_network = combine_subnetworks(self.name, individual, other_best_subnetworks)
            # Get the fitness from the game simulation
            self.fitness[i] = operators.evaluate_individual(env, full_network)

        # Find the best individual based on fitness
        self.best_individual = self.individuals[np.argmax(self.fitness)]


def combine_subnetworks(current_pop_name, current_individual, other_best_subnetworks):
    networks = [network for network in other_best_subnetworks]

    if current_pop_name == FEATURES_POP:
        return current_individual + networks

    if current_pop_name == WALK_LEFT_POP:
        return networks.insert(1, current_individual)

    if current_pop_name == WALK_RIGHT_POP:
        return networks.insert(2, current_individual)

    if current_pop_name == JUMP_POP:
        return networks.insert(3, current_individual)

    if current_pop_name == RELEASE_POP:
        return networks.insert(4, current_individual)

    return np.concatenate([current_individual, networks])


def evaluate_populations(env, subpopulations):
    num_subpopulations = len(subpopulations)

    # Iterate over each subpopulation
    for i in range(num_subpopulations):
        # Get the best individuals from other subpopulations
        other_best_subnetworks = []
        for j in range(num_subpopulations):
            if j != i:
                other_best_subnetworks.append(subpopulations[j].best_individual)

        # Evaluate all individuals in the current subpopulation
        subpopulations[i].evaluate(env, other_best_subnetworks)


def apply_evolutionary_operators(population, fitness):
    # @TODO: Implement this
    pass


def cooperative_coevolution(env, hidden_neurons):
    input_to_hidden_size = (env.get_num_sensors() + 1) * hidden_neurons
    hidden_to_output_size = hidden_neurons + 1

    features_pop = Subpopulation(FEATURES_POP, input_to_hidden_size)
    walk_left_pop = Subpopulation(WALK_LEFT_POP, hidden_to_output_size)
    walk_right_pop = Subpopulation(WALK_RIGHT_POP, hidden_to_output_size)
    jump_pop = Subpopulation(JUMP_POP, hidden_to_output_size)
    shoot_pop = Subpopulation(SHOOT_POP, hidden_to_output_size)
    release_pop = Subpopulation(RELEASE_POP, hidden_to_output_size)

    sub_populations = [features_pop, walk_left_pop, walk_right_pop, jump_pop, shoot_pop, release_pop]

    for generation in range(parameters.TOTAL_GENERATIONS):
        print(f"Generation {generation}")

        evaluate_populations(env, sub_populations)

        for subpop in sub_populations:
            subpop.individuals = apply_evolutionary_operators()

        for subpop in sub_populations:
            subpop.best_individual = subpop.individuals[np.argmax(subpop.fitness)]

    best_network = [subpop.best_individual for subpop in sub_populations]
    return np.hstack(best_network)
