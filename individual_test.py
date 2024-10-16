# imports
import numpy as np
import global_env
from evoman.environment import Environment

best_ind_file_path = 'train_ea1_4,6,7/best_ind_2.txt'

env = Environment(experiment_name=global_env.default_experiment_name,
                  multiplemode='no',
                  playermode=global_env.player_mode,
                  player_controller=global_env.player_controller,
                  enemymode=global_env.enemy_mode,
                  level=global_env.level,
                  speed=global_env.speed,
                  randomini='yes',
                  visuals=True)

best_solution = np.loadtxt(best_ind_file_path)

env.update_parameter('speed', 'normal')


total_wins = 0

for i in range(8):
    env.enemies = [i+1]

    print("\nTesting against enemy ", i+1, '...')

    _, player_life, enemy_life, time = env.play(pcont=best_solution)

    print(f"  Player Life: {player_life}")
    print(f"  Enemy Life: {enemy_life}")
    print(f"  Time: {time}")

    if enemy_life > player_life:
        print("  Result: Player Lost!")
    elif player_life > enemy_life:
        total_wins += 1
        print("  Result: Player Won!")
    else:
        print("  Result: It's a Draw!")

print("\nTotal wins: ", str(total_wins), " out of 8")

