import random 

# we need fitness to select parents:
def fitness(individual):
    # HERE: replace with our actual fitness evaluation
    return individual['fitness']

def tournament(population, k):
    selected_individuals = random.sample(population, k)
    best_individual = max(selected_individuals, key=fitness)
    return best_individual

# Selects 2 parents from the current population, choose k=3 or k=5
def parent_selection(population, k):
    p1 = tournament(population, k)
    p2 = tournament(population, k)
    return p1, p2

# Updates the population by selecting survivors.
def survivor_selection(population, offspring):
    pass
