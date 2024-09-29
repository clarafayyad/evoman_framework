import os

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def compute_stats(fitness_values):
    max_fitness_index = np.argmax(fitness_values)
    mean = np.mean(fitness_values)
    std = np.std(fitness_values)
    return max_fitness_index, mean, std

def extract_gains_from_file(algorithm_number, enemy_number):
    gains = []

    file_path = 'testing/test_ea' + str(algorithm_number) + '_e' + str(enemy_number) + '.csv'

    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            # Split the line by commas and extract the gain (last element)
            gain = float(line.strip().split(',')[-1])
            gains.append(gain)

    return gains

def statistical_test():
    # Extract gains for EA1 and EA2 from the specified enemies
    ea1_enemy1 = np.array(extract_gains_from_file(1, 1))
    ea2_enemy1 = np.array(extract_gains_from_file(2, 1))

    ea1_enemy7 = np.array(extract_gains_from_file(1, 7))
    ea2_enemy7 = np.array(extract_gains_from_file(2, 7))

    ea1_enemy8 = np.array(extract_gains_from_file(1, 8))
    ea2_enemy8 = np.array(extract_gains_from_file(2, 8))

    # Create a results list for storing the statistical test outcomes
    results = []

    # Perform Mann-Whitney U test for each enemy
    for enemy, ea1, ea2 in zip([1, 7, 8],
                               [ea1_enemy1, ea1_enemy7, ea1_enemy8],
                               [ea2_enemy1, ea2_enemy7, ea2_enemy8]):
        u_stat, p_value = stats.mannwhitneyu(ea1, ea2)
        significance = 'Significant difference' if p_value < 0.05 else 'No significant difference'
        results.append({'EA': f'EA1 vs EA2 (Enemy {enemy})', 'U-statistic': u_stat, 'p-value': p_value,
                        'Significance': significance})

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Print the results in a nice table format
    print("\nStatistical Test Results:")
    print(results_df)

    # Plot the table
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust margins
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=results_df.values,
                      colLabels=results_df.columns,
                      cellLoc='center',
                      loc='center')

    # Adjust table aesthetics
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # Scale the table for better visibility

    plt.title('Mann-Whitney U Test Results', fontsize=14, weight='bold')
    plt.savefig('graphs/statistical_test.png')
    plt.show()

