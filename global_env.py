from demo_controller import player_controller

# Experimental Setup
is_test = True
apply_dynamic_rewards = True
enemies = [1]
multiple_mode = 'no'
experiment_name = 'experiments'

# Tuner
tuner_trials = 100

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
