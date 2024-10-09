import numpy as np

def evaluate_individual(env, individual):
    """
     Evaluate an individual given an environment.
    :param env: Simulation Environment.
    :param individual: Numpy array of floats representing the individual.
    :return: float: Fitness value of the individual.
    """
    fitness, _, _, _ = env.play(pcont=individual)
    return fitness


def evaluate_population(env, population):
    """
    Evaluate a population given an environment.
    :param env: Simulation Environment
    :param population: 2D numpy array representing individuals in a population.
    :return: A numpy array representing the fitness values.
    """
    return np.array(list(map(lambda y: evaluate_individual(env, y), population)))

def player_health_reward(player_life):
    return player_life

def enemy_damage_reward(enemy_life):
    enemy_defeat_reward = 50
    base_reward = 100 - enemy_life
    if enemy_life == 0:
        return base_reward + enemy_defeat_reward
    return base_reward

def time_constraint_reward(enemy_health, time):
    max_time = 500
    time_normalized = time / max_time

    if enemy_health == 0:
        reward = 1 - time_normalized # reward faster enemy defeat
    else:
        reward = 0

    return max(0, min(reward, 1)) # between 0 and 1

def evaluate_objectives_and_fitness(env, individual):
    fitness, player_life, enemy_life, time = env.play(pcont=individual)

    player_health = player_health_reward(player_life)
    enemy_damage = enemy_damage_reward(enemy_life)
    time_reward = time_constraint_reward(enemy_life, time)

    return np.array([player_health, enemy_damage, time_reward]), fitness
