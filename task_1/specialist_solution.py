# imports
from specialist_test import test_experiment
from coevolution import cooperative_coevolution
from basic_evolution import basic_evolution

from demos.demo_controller import player_controller
from evoman.environment import Environment

from reporting import start_experiment, end_experiment
import time

# NN configuration
hidden_neurons = 10

# Set experiment name
experiment = 'experiments'
is_test = True
apply_coevolution = False
enemy_number = 1

# Initialize simulation
env = Environment(experiment_name=experiment,
                  enemies=[enemy_number],
                  playermode="ai",
                  player_controller=player_controller(hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes",
                  visuals=is_test)


ini_time = start_experiment(experiment, is_test)

if is_test:
    test_experiment(env, apply_coevolution, enemy_number)
else:
    if apply_coevolution:
        cooperative_coevolution(experiment, env, hidden_neurons)
    else:
        basic_evolution(experiment, env, hidden_neurons)

end_experiment(time.time() - ini_time)
