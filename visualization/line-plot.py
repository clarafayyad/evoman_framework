import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Function to read and aggregate multiple runs from CSV files
def aggregate_runs(data_folder, algorithm_name, pattern="train_results*.csv"):
    # Use glob to match all files in the folder based on the pattern
    files = glob.glob(os.path.join(data_folder, pattern))
    
    # List to store all dataframes
    dfs = []
    
    # Appends all data to the list
    for file in files:
        df = pd.read_csv(file)
        
        # Add a column identifying the algorithm (EA1 or EA2)
        df['algorithm'] = algorithm_name
        dfs.append(df)
    
    # Concatenate all dataframes along the row axis
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Group by 'gen' and 'algorithm' and aggregate by mean and standard deviation
    aggregated_data = all_data.groupby(['gen', 'algorithm']).agg({
        'best': ['mean', 'std'],
        'mean': ['mean', 'std'],
        'std': ['mean']
    }).reset_index()
    
    # Flatten the multi-level columns into individual columns
    aggregated_data.columns = ['gen', 'algorithm', 'best_mean', 'best_std', 'mean_fitness_mean', 'mean_fitness_std', 'std_mean']
    
    return aggregated_data

# Creating a line-plot across generations using average and standard deviation (std)
def plot_fitness_evolution(df, enemy_name, save_path):
    generations = df['gen'].unique()  # Unique generations
    
    # Plotting the mean fitness evolution with error bars
    plt.figure(figsize=(10, 6))
    
    # Plot for EA1
    df_ea1 = df[df['algorithm'] == 'EA1']
    plt.errorbar(df_ea1['gen'], df_ea1['mean_fitness_mean'], yerr=df_ea1['mean_fitness_std'], 
                 fmt='-o', label='EA1 Mean Fitness', color='lightskyblue', markersize=2, linewidth=0.5)
    plt.errorbar(df_ea1['gen'], df_ea1['best_mean'], yerr=df_ea1['best_std'], fmt='-o', 
                 label='EA1 Max Fitness', color='deepskyblue', markersize=3, linestyle='--', linewidth=0.5)

    # Plot for EA2
    df_ea2 = df[df['algorithm'] == 'EA2']
    plt.errorbar(df_ea2['gen'], df_ea2['mean_fitness_mean'], yerr=df_ea2['mean_fitness_std'], 
                 fmt='-s', label='EA2 Mean Fitness', color='peru', markersize=2, linewidth=0.5)
    plt.errorbar(df_ea2['gen'], df_ea2['best_mean'], yerr=df_ea2['best_std'], fmt='-s', 
                 label='EA2 Max Fitness', color='chocolate', markersize=3, linestyle='--', linewidth=0.5)

    # Labels and title
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Fitness', fontsize=16)
    plt.title(f'Fitness Evolution across Generations (Enemy: {enemy_name})', fontsize=20)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(save_path, dpi=300)
    
    # Show the plot
    plt.show()

# Set the folders containing the data files for EA1 and EA2
data_folder_ea1 = 'train_ea1_e1'  # Folder for EA1
data_folder_ea2 = 'train_ea1_e7'  # Folder for EA2

enemy_name = 'Enemy_1'

# Save path using the enemy name
save_path = f'results/line-plots/aggregated_fitness_evolution_{enemy_name.lower()}.png'

# Call the function to aggregate data for EA1 and EA2
df_aggregated_ea1 = aggregate_runs(data_folder_ea1, 'EA1')
df_aggregated_ea2 = aggregate_runs(data_folder_ea2, 'EA2')

# Combine the aggregated data from EA1 and EA2
df_aggregated = pd.concat([df_aggregated_ea1, df_aggregated_ea2], ignore_index=True)

# Plot the combined data
plot_fitness_evolution(df_aggregated, enemy_name, save_path)

print(f"Aggregated plot saved as {save_path}")