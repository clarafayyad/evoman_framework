import csv
import optuna
import global_env
from evoman.environment import Environment
from ea_config import EAConfigs
from coevolution import CoevolutionaryAlgorithm

def objective(trial):
    population_size = trial.suggest_int("population_size", 50, 1000)
    total_generations = trial.suggest_int("total_generations", 50, 100)
    tournament_size = trial.suggest_int("tournament_size", 3, 20)
    mutation_rate = trial.suggest_float("mutation_rate", 0.1, 0.9)
    mutation_sigma = trial.suggest_float("mutation_sigma", 0.1, 0.9)
    selection_pressure = trial.suggest_float("selection_pressure", 1.0, 2.0)
    crossover_rate = trial.suggest_float("crossover_rate", 0.1, 0.9)
    crossover_weight = trial.suggest_float("crossover_weight", 0.1, 0.9)

    configs = EAConfigs(population_size, total_generations,
        tournament_size, mutation_rate, mutation_sigma,
        selection_pressure, crossover_weight, crossover_rate)
    run_ea(configs)
    return fetch_max_fitness()

def run_ea(configs):
    env = Environment(experiment_name=global_env.experiment_name,
                      enemies=global_env.enemies,
                      multiplemode=global_env.multiple_mode,
                      playermode=global_env.player_mode,
                      player_controller=global_env.player_controller,
                      enemymode=global_env.enemy_mode,
                      level=global_env.level,
                      speed=global_env.speed,
                      randomini=global_env.random_ini,
                      visuals=global_env.visuals)
    co_evolutionary_EA = CoevolutionaryAlgorithm(configs)
    co_evolutionary_EA.multi_obj_eval = global_env.apply_multi_objective
    co_evolutionary_EA.cooperative_coevolution(env)

def fetch_max_fitness():
    with open('experiments/train_results.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        last_row = None
        for row in csv_reader:
            last_row = row
        if last_row:
            return float(last_row[1])
        return None

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=global_env.tuner_trials)
    print(f"Best parameters: {study.best_params}")
    print(f"Best fitness: {study.best_value}")
