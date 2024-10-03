# imports
import global_env
from coevolution import CoevolutionaryAlgorithm
from ea_config import EAConfigs

from evoman.environment import Environment


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
    configs = EAConfigs()
    co_evolutionary_EA = CoevolutionaryAlgorithm(configs)
    co_evolutionary_EA.cooperative_coevolution(env)
