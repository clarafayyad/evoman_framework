# EvoMan Framework
Evoman is a video game playing framework to be used as a testbed for optimization algorithms.
A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

# Simulation Training and Testing

## Overview

This project implements a neural network-based agent for a game simulation using evolutionary algorithms. The agent can be trained or tested using the provided script `specialist_solution.py`.

## Instructions

To train or test the agent, follow these steps:

1. **Open `specialist_solution.py`.**
2. Set the following variables according to your needs:

   - `is_test`: 
     - Set to `True` if you want to test the agent with previously trained weights.
     - Set to `False` if you want to train the agent.

   - `apply_coevolution`: 
     - Set to `True` if you want to apply the cooperative coevolutionary algorithm.
     - Set to `False` if you prefer the regular evolutionary algorithm.

   - `enemy_number`: 
     - Set the enemy number (1-8) for the specific game enemy you want to train/test against.
3. Run `specialist_solution.py`.
