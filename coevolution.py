import numpy as np
import operators
import reporting

# Define a set of constants
POPULATION_SIZE = 100
TOTAL_GENERATIONS = 50

# Set EA Operators Parameters
lower_bound = -1
upper_bound = 1
tournament_size = 5
mutation_rate = 0.6
mutation_sigma = 0.3
selection_pressure = 1
crossover_weight = 0.8
crossover_rate = 0.6
sigma_share = 0.8  # niche radius

# Define subnetworks
FEATURES_POP = 'input_to_hidden'
WALK_LEFT_POP = 'walk left'
WALK_RIGHT_POP = 'walk right'
JUMP_POP = 'jump'
SHOOT_POP = 'shoot'
RELEASE_POP = 'release'


class Subpopulation:
    def __init__(self, identifier, individuals):
        # Initialize with random individuals
        self.individuals = individuals
        # Store fitness for each individual
        self.fitness = np.zeros(len(individuals))
        # Select a random individual as the initial best individual
        self.best_individual = self.individuals[np.random.randint(len(individuals))]
        self.identifier = identifier

    def evaluate(self, env, other_best_subnetworks):
        for i, individual in enumerate(self.individuals):
            # Combine current individual with the best individuals from the other subnetworks
            full_network = combine_subnetworks(self.identifier, individual, other_best_subnetworks)
            # Get the fitness from the game simulation
            self.fitness[i] = operators.evaluate_individual(env, full_network)

        # Find the best individual based on fitness
        self.best_individual = self.individuals[np.argmax(self.fitness)]

    def evolve(self, env, other_best_subnetworks):
        # Evaluate current subpopulation
        self.evaluate(env, other_best_subnetworks)

        # Create Offspring
        tournament_count = int(len(self.individuals) / 2)
        offspring = operators.arithmetic_uniform_crossover(
            self.individuals,
            self.fitness,
            tournament_count,
            tournament_size,
            crossover_weight,
            crossover_rate,
        )

        # Mutate offspring
        for i in range(len(offspring)):
            # Apply gaussian mutation
            offspring[i] = operators.gaussian_mutation(
                offspring[i],
                rate=mutation_rate,
                sigma=mutation_sigma
            )
            # Clamp the weights and biases within the initial range after applying variation operators
            operators.clamp_within_bounds(offspring[i], lower_bound, upper_bound)

        # Evaluate offspring
        offspring_sub_pop = Subpopulation(self.identifier, offspring)
        offspring_sub_pop.evaluate(env, other_best_subnetworks)

        # Apply fitness sharing
        # fitness, offspring_fitness = operators.fitness_sharing(
        #     self.individuals,
        #     self.fitness,
        #     offspring_sub_pop.individuals,
        #     offspring_sub_pop.fitness,
        #     parameters.sigma_share
        # )
        # self.fitness = fitness
        # offspring_sub_pop.fitness = offspring_fitness

        # Survivor selection
        selected_individuals, selected_fitness_values = operators.linear_ranking_survivor_selection(
            self.individuals,
            self.fitness,
            offspring_sub_pop.individuals,
            offspring_sub_pop.fitness,
            s=selection_pressure
        )

        self.individuals = selected_individuals
        self.fitness = selected_fitness_values
        self.best_individual = self.individuals[np.argmax(self.fitness)]


def combine_subnetworks(current_pop_id, current_individual, other_best_subnetworks):
    network_order = [FEATURES_POP, WALK_LEFT_POP, WALK_RIGHT_POP, JUMP_POP, SHOOT_POP, RELEASE_POP]

    network_parts = {current_pop_id: current_individual}
    for identifier, individual in other_best_subnetworks.items():
        network_parts[identifier] = individual

    combined_network = []
    for key in network_order:
        combined_network.append(network_parts[key])

    return np.hstack(combined_network)


def initialize_random_sub_population(identifier, individual_size):
    return Subpopulation(identifier, operators.initialize_population(
        POPULATION_SIZE,
        individual_size,
        lower_bound,
        upper_bound
    ))


def cooperative_coevolution(experiment, env, hidden_neurons):
    input_to_hidden_size = (env.get_num_sensors() + 1) * hidden_neurons
    hidden_to_output_size = hidden_neurons + 1

    # Initialize subpopulations
    features_pop = initialize_random_sub_population(FEATURES_POP, input_to_hidden_size)
    walk_left_pop = initialize_random_sub_population(WALK_LEFT_POP, hidden_to_output_size)
    walk_right_pop = initialize_random_sub_population(WALK_RIGHT_POP, hidden_to_output_size)
    jump_pop = initialize_random_sub_population(JUMP_POP, hidden_to_output_size)
    shoot_pop = initialize_random_sub_population(SHOOT_POP, hidden_to_output_size)
    release_pop = initialize_random_sub_population(RELEASE_POP, hidden_to_output_size)

    subpopulations = [features_pop, walk_left_pop, walk_right_pop, jump_pop, shoot_pop, release_pop]
    subpopulations_len = len(subpopulations)

    best_individual_found = None
    best_fitness_found = 0

    # Co-evolution
    for generation in range(TOTAL_GENERATIONS):
        # Evolve each subpopulation
        for i in range(subpopulations_len):
            # Get the best individuals from the other subpopulations
            other_best_subnetworks = {}
            for j in range(subpopulations_len):
                if j != i:
                    other_best_subnetworks[subpopulations[j].identifier] = subpopulations[j].best_individual

            # Evolve current subpopulation by evaluating it with the best individuals from the other subpopulations
            subpopulations[i].evolve(env, other_best_subnetworks)

        # Create best network out of subpopulations
        current_best_network = np.hstack([subpop.best_individual for subpop in subpopulations])
        current_best_fitness = operators.evaluate_individual(env, current_best_network)
        reporting.log_stats(experiment, generation, current_best_fitness, 0, 0)

        if current_best_fitness > best_fitness_found:
            best_individual_found = current_best_network
            best_fitness_found = current_best_fitness

    reporting.save_best_individual(experiment, best_individual_found, best_fitness_found)





