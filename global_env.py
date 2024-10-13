from demo_controller import player_controller

# Experimental Setup
is_test = False
apply_dynamic_rewards = True
enemies = [4, 6, 7]
multiple_mode = 'yes'
default_experiment_name = 'experiments'

# Training
training_runs = 10

# Tuner
tuner_trials = 30

# NN configuration
hidden_neurons = 10
lower_bound = -1
upper_bound = 1

# Environment Setup
player_mode = 'ai'
player_controller = player_controller(hidden_neurons)
enemy_mode = 'static'
level = 2
speed = 'fastest'
visuals = is_test
random_ini = 'no'
if is_test:
    random_ini = 'yes'
