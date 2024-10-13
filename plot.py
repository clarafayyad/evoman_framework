import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Set up a list to collect all data
data = []

# Read all 10 files
for file in glob.glob('train_ea2_4,6,7/train_results_*.csv'):
    df = pd.read_csv(file)
    data.append(df)

# Concatenate all data into one DataFrame
combined_data = pd.concat(data)

# Group by generation to aggregate mean and std across files
aggregated_data = combined_data.groupby('gen').agg({
    'best': ['mean'],
    'mean': ['mean', 'std'],
    'std': ['mean']
}).reset_index()

# Rename the columns for easier plotting
aggregated_data.columns = ['gen', 'best_mean', 'mean_mean', 'mean_std', 'std_mean']

# Plotting
plt.figure(figsize=(10,6))
plt.plot(aggregated_data['gen'], aggregated_data['best_mean'], label='Best Fitness (Mean)', color='blue')
plt.plot(aggregated_data['gen'], aggregated_data['mean_mean'], label='Mean Fitness', color='green')
plt.fill_between(aggregated_data['gen'],
                 aggregated_data['mean_mean'] - aggregated_data['mean_std'],
                 aggregated_data['mean_mean'] + aggregated_data['mean_std'],
                 color='green', alpha=0.2, label='Mean Std Dev')

plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()
