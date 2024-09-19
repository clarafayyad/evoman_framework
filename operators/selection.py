import random 


def fitness(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def tournament(population, k, env):
    selected_individuals = random.sample(population, k)
    # Find the best individual among the selected based on fitness
    best_individual = max(selected_individuals, key=lambda ind: fitness(env, ind['pcont']))
    
    return best_individual

# Selects 2 parents from the current population, choose k=3 or k=5
def parent_selection(population, k, env):
    p1 = tournament(population, k, env)
    p2 = tournament(population, k, env)
    return p1, p2

# Updates the population by selecting survivors.
def survivor_selection(population, offspring):
    pass
