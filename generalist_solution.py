# imports
import time

import dynamic_rewards_hyperparams
import global_env
import hyperparams
from ea_config import EAHyperParams
from generalist_test import test_experiment
from evoman.environment import Environment
from reporting import start_experiment, end_experiment
from basic_evolution import BasicEvolutionaryAlgorithm

# Initialize simulation
env = Environment(experiment_name=global_env.default_experiment_name,
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
    ini_time = start_experiment(global_env.default_experiment_name, global_env.is_test)

    if global_env.is_test:
        test_experiment(env)
    else:
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
        ea = BasicEvolutionaryAlgorithm(params)
        ea.experiment = global_env.default_experiment_name
        ea.execute_evolution(env)

    end_experiment(time.time() - ini_time)

