import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def aggregate_data(folder_pattern):
    """Function to aggregate data from a folder based on the file pattern."""
    data = []
    for file in glob.glob(folder_pattern):  # Read all files matching the pattern in the folder
        df = pd.read_csv(file)
        data.append(df)

    # Concatenate all data into one DataFrame
    combined_data = pd.concat(data)

    # Remove rows where generation > 100
    combined_data = combined_data[combined_data['gen'] <= 100]

    # Group by generation to aggregate mean and std across files
    aggregated_data = combined_data.groupby('gen').agg({
        'best': ['mean'],
        'mean': ['mean', 'std'],
        'std': ['mean']
    }).reset_index()

    # Rename the columns for easier plotting
    aggregated_data.columns = ['gen', 'best_mean', 'mean_mean', 'mean_std', 'std_mean']

    return aggregated_data


# Aggregate data from both folders
folder1_data = aggregate_data('train_ea1_3,7,8/train_*.csv')  # Modify 'folder1' with the actual folder path
folder2_data = aggregate_data('train_ea2_3,7,8/train_*.csv')  # Modify 'folder2' with the actual folder path

# Plotting
plt.figure(figsize=(10, 6))

# Plot for folder 1
plt.plot(folder1_data['gen'], folder1_data['best_mean'], label='EA1 Best Fitness', color='DodgerBlue')
plt.plot(folder1_data['gen'], folder1_data['mean_mean'], label='EA1 Mean', color='navy')
plt.fill_between(folder1_data['gen'],
                 folder1_data['mean_mean'] - folder1_data['mean_std'],
                 folder1_data['mean_mean'] + folder1_data['mean_std'],
                 color='royalblue', alpha=0.2, label='EA1 Std')

# Plot for folder 2
plt.plot(folder2_data['gen'], folder2_data['best_mean'], label='EA2 Best Fitness', color='orangered', linestyle='--')
plt.plot(folder2_data['gen'], folder2_data['mean_mean'], label='EA2 Mean', color='brown', linestyle='--')
plt.fill_between(folder2_data['gen'],
                 folder2_data['mean_mean'] - folder2_data['mean_std'],
                 folder2_data['mean_mean'] + folder2_data['mean_std'],
                 color='coral', alpha=0.2, label='EA2 Std')

# Customize the plot
plt.title('Enemy group: [3,7,8]')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)

plt.savefig('graphs/line_plot_enemies_' + ','.join(map(str, [3,7,8])) + '.png', dpi=300)

plt.show()
