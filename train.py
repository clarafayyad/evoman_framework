import os

import dynamic_rewards_hyperparams
import global_env
import hyperparams
from basic_evolution import BasicEvolutionaryAlgorithm
from evoman.environment import Environment
from ea_config import EAHyperParams


def get_experiment_name():
    name = 'train_'
    if global_env.apply_dynamic_rewards:
        name += 'ea2_'
    else:
        name += 'ea1_'
    return name + ','.join(map(str, global_env.enemies))


experiment = get_experiment_name()

# Create experiment folder if it doesn't exist
os.makedirs(experiment, exist_ok=True)

env = Environment(experiment_name=experiment,
                  enemies=global_env.enemies,
                  multiplemode=global_env.multiple_mode,
                  playermode=global_env.player_mode,
                  player_controller=global_env.player_controller,
                  enemymode=global_env.enemy_mode,
                  level=global_env.level,
                  speed=global_env.speed,
                  randomini=global_env.random_ini,
                  visuals=global_env.is_test)

population_size = hyperparams.population_size
total_generations = hyperparams.total_generations
tournament_size = hyperparams.tournament_size
mutation_rate = hyperparams.mutation_rate
mutation_sigma = hyperparams.mutation_sigma
selection_pressure = hyperparams.selection_pressure
crossover_weight = hyperparams.crossover_weight
crossover_rate = hyperparams.crossover_rate
if global_env.apply_dynamic_rewards:
    population_size = dynamic_rewards_hyperparams.population_size
    total_generations = dynamic_rewards_hyperparams.total_generations
    tournament_size = dynamic_rewards_hyperparams.tournament_size
    mutation_rate = dynamic_rewards_hyperparams.mutation_rate
    mutation_sigma = dynamic_rewards_hyperparams.mutation_sigma
    selection_pressure = dynamic_rewards_hyperparams.selection_pressure
    crossover_weight = dynamic_rewards_hyperparams.crossover_weight
    crossover_rate = dynamic_rewards_hyperparams.crossover_rate

params = EAHyperParams(
    population_size=population_size,
    total_generations=total_generations,
    tournament_size=tournament_size,
    mutation_rate=mutation_rate,
    mutation_sigma=mutation_sigma,
    selection_pressure=selection_pressure,
    crossover_weight=crossover_weight,
    crossover_rate=crossover_rate,
)

for i in range(global_env.training_runs):
    ea = BasicEvolutionaryAlgorithm(params)
    ea.experiment = experiment
    ea.run_number = i
    ea.execute_evolution(env)
