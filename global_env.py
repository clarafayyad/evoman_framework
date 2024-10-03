from demo_controller import player_controller

# NN configuration
hidden_neurons = 10

# Set global env specs
is_test = False
experiment_name = 'experiments'
enemies = [1]
multiple_mode = "no"
player_mode = "ai"
player_controller = player_controller(hidden_neurons)
enemy_mode = "static"
level = 2
speed = "fastest"

random_ini = "no"
if is_test:
    random_ini = "yes"

visuals = is_test