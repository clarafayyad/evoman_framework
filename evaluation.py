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


landscape_difficulty_rewards = {
    1: 0.5, # Stairs, moderate complexity
    2: 0, # Flat surface, simplest
    3: 0, # Flat surface, simplest
    4: 0, # Flat surface, simplest
    5: 0, # Flat surface, simplest
    6: 0, # Flat surface, simplest
    7: 1, # Water, most complex
    8: 0.5, # Stairs, moderate complexity
}

action_difficulty_rewards = {
    1: 0.5, # 4 actions, moderate complexity
    2: 0.5, # 4 actions, moderate complexity
    3: 0.5, # 4 actions, moderate complexity
    4: 0.5, # 4 actions, moderate complexity
    5: 0, # 3 actions, simplest
    6: 0, # 3 actions, simplest
    7: 1, # 6 actions, most complex
    8: 1, # 6 actions, most complex
}

def enemy_strength_reward(enemies):
    enemy_landscape_rewards = np.array([landscape_difficulty_rewards[enemy] for enemy in enemies])
    enemy_actions_rewards = np.array([action_difficulty_rewards[enemy] for enemy in enemies])
    mean_landscape_reward = np.mean(enemy_landscape_rewards)
    mean_actions_reward = np.mean(enemy_actions_rewards)
    return 0.5 * mean_landscape_reward + 0.5 * mean_actions_reward

def player_health_reward(player_life):
    return player_life

def enemy_damage_reward(enemy_life):
    enemy_defeat_reward = 50
    base_reward = 100 - enemy_life
    if enemy_life == 0:
        return base_reward + enemy_defeat_reward
    return base_reward

def time_constraint_reward(player_health, enemy_health, time):
    max_time = 500
    time_normalized = time / max_time

    if player_health == 0:
        reward = min(time_normalized, 1) # reward longer survival
    elif enemy_health == 0:
        reward = 1 - time_normalized # reward faster enemy defeat
    else:
        reward = 0.5

    return max(0, min(reward, 1)) # between 0 and 1

def reward_objectives(env, individual):
    fitness, player_life, enemy_life, time = env.play(pcont=individual)

    enemy_strength = enemy_strength_reward(env.enemies)
    player_health = player_health_reward(player_life)
    enemy_damage = enemy_damage_reward(enemy_life)
    time_reward = time_constraint_reward(player_life, enemy_life, time)

    return np.array([enemy_strength, player_health, enemy_damage, time_reward])

def select_best_pareto_individual(fitness):
    ideal_point = np.max(fitness, axis=0)
    distances = np.linalg.norm(fitness - ideal_point, axis=1)
    best_index = np.argmin(distances)
    return best_index