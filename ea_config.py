class EAConfigs:
    def __init__(self, population_size, total_generations,
                 tournament_size, mutation_rate, mutation_sigma,
                 selection_pressure, crossover_weight, crossover_rate):
        self.population_size = population_size
        self.total_generations = total_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.selection_pressure = selection_pressure
        self.crossover_weight = crossover_weight
        self.crossover_rate = crossover_rate

    def display_config(self):
        print(f"Population Size: {self.population_size}")
        print(f"Total Generations: {self.total_generations}")
        print(f"Tournament Size: {self.tournament_size}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Mutation Sigma: {self.mutation_sigma}")
        print(f"Selection Pressure: {self.selection_pressure}")
        print(f"Crossover Weight: {self.crossover_weight}")
        print(f"Crossover Rate: {self.crossover_rate}")

