# imports
import time

import global_env
import hyperparams
import multi_obj_hyperparams
from coevolution import CoevolutionaryAlgorithm
from multi_obj_coevolution import CoevolutionaryMultiObjAlgorithm
from ea_config import EAConfigs
from generalist_test import test_experiment
from evoman.environment import Environment
from reporting import start_experiment, end_experiment

# Initialize simulation
env = Environment(experiment_name=global_env.experiment_name,
                  enemies=global_env.enemies,
                  multiplemode=global_env.multiple_mode,
                  playermode=global_env.player_mode,
                  player_controller=global_env.player_controller,
                  enemymode=global_env.enemy_mode,
                  level=global_env.level,
                  speed=global_env.speed,
                  randomini=global_env.random_ini,
                  visuals=global_env.is_test)

if __name__ == "__main__":
    ini_time = start_experiment(global_env.experiment_name, global_env.is_test)

    if global_env.is_test:
        test_experiment(env)
    else:
        if global_env.apply_multi_objective:
            configs = EAConfigs(
                population_size=multi_obj_hyperparams.population_size,
                total_generations=multi_obj_hyperparams.total_generations,
                tournament_size=multi_obj_hyperparams.tournament_size,
                mutation_rate=multi_obj_hyperparams.mutation_rate,
                mutation_sigma=multi_obj_hyperparams.mutation_sigma,
                selection_pressure=0,  # unused here
                crossover_weight=multi_obj_hyperparams.crossover_weight,
                crossover_rate=multi_obj_hyperparams.crossover_rate,
            )
            ea = CoevolutionaryMultiObjAlgorithm(configs)
            ea.cooperative_coevolution(env)
        else:
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
            ea = CoevolutionaryAlgorithm(configs)
            ea.cooperative_coevolution(env)

    end_experiment(time.time() - ini_time)
