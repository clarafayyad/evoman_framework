# EvoMan Framework
Evoman is a video game playing framework to be used as a testbed for optimization algorithms.
A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


# Overview

This project implements a neural network-based agent for a game simulation using evolutionary algorithms. 

## Training

To train the agent, follow these steps:

1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `is_test`: `False`
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `apply_multi_objective`: 
     - Set to `True` if you want to apply the multi objective evaluation algorithm.
     - Set to `False` if you prefer the regular co-evolutionary algorithm.
   - `enemies`: Set as a list of enemies (1-8) to train the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are training against more than one enemy. 
     - Set to `"no"` if you are training against one enemy only.
3. Run `generalist_solution.py`.
4. The stats will be generated under the [experiments](/experiments) folder, and the best solution will be saved in [this file](experiments/best.txt).  

## Testing 

To test the agent, follow these steps:

1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `is_test`: `True`
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `apply_multi_objective`: 
     - Set to `True` if you want to apply the multi objective evaluation algorithm.
     - Set to `False` if you prefer the regular co-evolutionary algorithm.
   - `enemies`: Set as a list of enemies (1-8) to test the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are testing against more than one enemy. 
     - Set to `"no"` if you are testing against one enemy only.
3. Run `generalist_solution.py`.
4. The output of the test will be printed in the console. 


## Tuning the Hyperparameters

- The hyperparameters of the regular co-evolutionary algorithm can be found and changed in [the hyperparameters file](hyperparams.py). 
- The hyperparameters of the multi-objective co-evolutionary algorithm can be found and changed [the multi-obj hyperparameters file](multi_obj_hyperparams.py).

To tune the algorithm using optuna, follow these steps: 
1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `tuner_trials`: This is the number of trials the tuner will run to find the best parameters. 
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `apply_multi_objective`: 
     - Set to `True` if you want to apply the multi objective evaluation algorithm.
     - Set to `False` if you prefer the regular co-evolutionary algorithm.
   - `enemies`: Set as a list of enemies (1-8) to tune the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are tuning against more than one enemy. 
     - Set to `"no"` if you are tuning against one enemy only.
2. Run `tuner.py`
3. The best found parameters will be printed in the console and can be used to update the parameter files. 