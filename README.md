# EvoMan Framework
Evoman is a video game playing framework to be used as a testbed for optimization algorithms.
A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI


# Overview

This project implements a neural network-based agent for a game simulation using evolutionary algorithms. 

## Experimenting
To experiment and train the agent without saving results for multiple runs, follow these steps:

1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `is_test`: `False`
   - `enemies`: Set as a list of enemies (1-8) to train the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are training against more than one enemy. 
     - Set to `"no"` if you are training against one enemy only.
3. Run `generalist_solution.py`.
4. Stats and best individual will be generated and saved under [the experiments folder](/experiments). This folder will be overwritten everytime you run this file.  


## Training

To train the agent, follow these steps:

1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `is_test`: `False`
   - `training_runs`: Set as number of times you want to train the algorithm (default is 10)
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `enemies`: Set as a list of enemies (1-8) to train the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are training against more than one enemy. 
     - Set to `"no"` if you are training against one enemy only.
3. Run `train.py`.
4. A folder will be created depending on the algorithm chosen and enemies _(ex: train_ea1_4,6,7)_. Per run, two files will be generated under that folder (train_results_i_.csv, and best_ind_i.txt).

## Testing 

### To test the agent against a specific enemy, follow these steps:

1. Open [the individual test file](individual_test.py), and set `best_ind_file_path`: Set as the file path to the player you want to test with _(e.g. 'experiments/best_ind_0.txt')_
2. Run `individual_test.py`.


### To test a trained agent against enemy groups:
1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `is_test`: `True`
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `enemies`: Set as a list of enemies (1-8) to test the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are testing against more than one enemy. 
     - Set to `"no"` if you are testing against one enemy only.
3. Run `generalist_solution.py`.
4. The output of the test will be printed in the console. 


## Tuning the Hyperparameters

- The hyperparameters of the regular evolutionary algorithm can be found and changed in [this hyperparameters file](hyperparams.py). 
- The hyperparameters of the dynamic rewards evolutionary algorithm can be found and changed [this hyperparameters file](dynamic_rewards_hyperparams.py).

To tune the algorithm using optuna, follow these steps: 
1. Open [the global env file](global_env.py), and set the following variables according to your needs:
   - `tuner_trials`: This is the number of trials the tuner will run to find the best parameters. 
   - `apply_dynamic_rewards`: 
     - Set to `True` if you want to apply the dynamic rewards' evaluation.
     - Set to `False` if you prefer the regular evaluation.
   - `enemies`: Set as a list of enemies (1-8) to tune the agent against. 
   - `multiple_mode`:
     - Set to `"yes"` if you are tuning against more than one enemy. 
     - Set to `"no"` if you are tuning against one enemy only.
2. Run `tuner.py`
3. The best found parameters will be printed in the console and can be used to update the parameter files. 