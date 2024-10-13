import numpy as np

import global_env


def basic_evaluation(env, individual):
    env.fitness_player_health_weight = 0.1
    env.fitness_enemy_damage_weight = 0.9
    env.fitness_time_weight = 1
    fitness, _, _, _ = env.play(pcont=individual)
    return fitness


def evaluate_individual(env, generation, total_generations, individual):
    """
     Evaluate an individual given an environment.
    :param env: Simulation Environment.
    :param generation: current generation number of the evolution process
    :param total_generations: total number of generations of the evolution process
    :param individual: Numpy array of floats representing the individual.
    :return: float: Fitness value of the individual.
    """
    if global_env.apply_dynamic_rewards:
        return dynamic_evaluation(env, generation, total_generations, individual)
    return basic_evaluation(env, individual)


def dynamic_evaluation(env, generation, total_generations, individual):
    # define 3 phases
    phase_player_health_end = total_generations / 3
    phase_enemy_damage_end = total_generations / 3 * 2
    phase_time_constraint_end = total_generations

    # Initialize weights
    if generation <= phase_player_health_end:
        env.fitness_player_health_weight = 0.8
        env.fitness_enemy_damage_weight = 0.2
        env.fitness_time_weight = 0.0
    elif generation <= phase_enemy_damage_end:
        env.fitness_player_health_weight = 0.1
        env.fitness_enemy_damage_weight = 0.9
        env.fitness_time_weight = 0.0
    elif generation <= phase_time_constraint_end:
        env.fitness_player_health_weight = 0.1
        env.fitness_enemy_damage_weight = 0.9
        env.fitness_time_weight = 1.0
    else:
        # Default
        env.fitness_player_health_weight = 0.1
        env.fitness_enemy_damage_weight = 0.9
        env.fitness_time_weight = 1

    fitness, _, _, _ = env.play(pcont=individual)
    return fitness

def evaluate_population(env, generation, total_generations, population, force_basic_eval=False):
    """
    Evaluate a population given an environment.
    :param env: Simulation Environment
    :param generation: current generation number of the evolution process
    :param total_generations: total number of generations of the evolution process
    :param population: 2D numpy array representing individuals in a population.
    :param force_basic_eval: force basic evaluation of the population. (non-dynamic evaluation)
    :return: A numpy array representing the fitness values.
    """
    if force_basic_eval:
        return np.array(list(map(lambda y: basic_evaluation(env, y), population)))
    return np.array(list(map(lambda y: evaluate_individual(env, generation, total_generations, y), population)))

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
        reward = 1 - time_normalized  # reward faster enemy defeat
    else:
        reward = 0

    return max(0, min(reward, 1))  # between 0 and 1


def evaluate_objectives_and_fitness(env, individual):
    fitness, player_life, enemy_life, time = env.play(pcont=individual)

    player_health = player_health_reward(player_life)
    enemy_damage = enemy_damage_reward(enemy_life)
    time_reward = time_constraint_reward(enemy_life, time)

    return np.array([player_health, enemy_damage, time_reward]), fitness
