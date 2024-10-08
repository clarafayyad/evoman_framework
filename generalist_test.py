# imports
import numpy as np
import global_env


# loads file with the best solution for testing
def test_experiment(env):
    best_solution = np.loadtxt(global_env.experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    _, player_life, enemy_life, time = env.play(pcont=best_solution)

    print(f"\nPlayer Life: {player_life}")
    print(f"Enemy Life: {enemy_life}")
    print(f"Time: {time}")
    if enemy_life > player_life:
        print("Result: Player Lost!")
    elif player_life > enemy_life:
        print("Result: Player Won!")
    else:
        print("Result: It's a Draw!")

