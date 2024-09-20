# imports
import optimization_specialist_test
import optimization_specialist_train
from demo_controller import player_controller
from evoman.environment import Environment
import reporting
import time

# NN configuration
hidden_neurons = 10

# Set experiment name
experiment = 'experiments'

# Initialize simulation
env = Environment(experiment_name=experiment,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

is_test = False
ini_time = reporting.start_experiment(experiment, is_test)

if is_test:
    optimization_specialist_test.test_experiment(experiment, env)
else:
    optimization_specialist_train.train_experiment(experiment, env, hidden_neurons)

reporting.end_experiment(time.time() - ini_time)
