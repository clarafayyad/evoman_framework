import os

import global_env
import hyperparams
from ea_config import EAConfigs
from basic_evolution import BasicEvolutionaryAlgorithm
from evoman.environment import Environment


def get_experiment_name():
    return 'train_ea1_' + ','.join(map(str, global_env.enemies))


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

configs = EAConfigs(
    population_size=hyperparams.population_size,
    total_generations=hyperparams.total_generations,
    tournament_size=hyperparams.tournament_size,
    mutation_rate=hyperparams.mutation_rate,
    mutation_sigma=hyperparams.mutation_sigma,
    selection_pressure=hyperparams.selection_pressure,
    crossover_weight=hyperparams.crossover_weight,
    crossover_rate=hyperparams.crossover_rate,
)

for i in range(global_env.training_runs):
    ea = BasicEvolutionaryAlgorithm(configs)
    ea.experiment = experiment
    ea.run_number = i
    ea.execute_evolution(env)
