class EAConfigs:
    def __init__(self, population_size=100, total_generations=50,
                 tournament_size=3, mutation_rate=0.8, mutation_sigma=0.5,
                 selection_pressure=1.1, crossover_weight=0.8, crossover_rate=0.8):
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

