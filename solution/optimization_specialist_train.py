# imports
from evoman.environment import Environment
import time
import os

# initialize simulation
env = Environment()

# check environment state
env.state_to_log()

# set time marker
ini = time.time()

# initialize population loading old solutions or generating new ones
experiment_name = ''
if not os.path.exists(experiment_name + '/evoman_solstate'):
    print('\nNEW EVOLUTION\n')
    # generate new population here
else:
    print('\nCONTINUING EVOLUTION\n')
    # generate another population here

# save results in a txt file and output/print some relevant messages
# code here

# evolution
condition_to_stop = False
while not condition_to_stop:
    # parent selection
    # create offspring
    # evaluate offspring
    # survivor selection

    # save results in the txt file and output relevant msgs
    # save file with the best solution
    # save simulation state
    break

# print total execution time for experiment

# save control (simulation has ended) file for bash loop file

# check environment state
