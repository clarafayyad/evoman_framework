import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Function to read and aggregate multiple runs from CSV files
def aggregate_runs(data_folder, algorithm_name, pattern="train_result*.csv"):
    files = glob.glob(os.path.join(data_folder, pattern))
    dfs = []
    
    for file in files:
        df = pd.read_csv(file)
        df['algorithm'] = algorithm_name
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    aggregated_data = all_data.groupby(['gen', 'algorithm']).agg({
        'best': ['mean', 'std'],
        'mean': ['mean', 'std'],
        'std': ['mean']
    }).reset_index()
    
    aggregated_data.columns = ['gen', 'algorithm', 'best_mean', 'best_std', 'mean_fitness_mean', 'mean_fitness_std', 'std_mean']
    return aggregated_data

# Function to plot EA2 subsections
def plot_ea2_subsection(df, enemy_name, save_path, label_suffix, color_mean, color_max):
    plt.errorbar(df['gen'], df['mean_fitness_mean'], yerr=df['mean_fitness_std'], 
                fmt='-s', label=f'EA2 Mean Fitness {label_suffix}', color=color_mean, markersize=2, linewidth=0.5)
    plt.errorbar(df['gen'], df['best_mean'], yerr=df['best_std'], fmt='--s', 
                label=f'EA2 Max Fitness {label_suffix}', color=color_max, markersize=3, linewidth=0.5)


# Creating a line-plot across generations using average and standard deviation (std)
def plot_fitness_evolution(df, enemy_name, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot for EA1
    df_ea1 = df[df['algorithm'] == 'EA1']
    df_ea1_50 = df_ea1[df_ea1['gen'] <= 49]
    plt.errorbar(df_ea1_50['gen'], df_ea1_50['mean_fitness_mean'], yerr=df_ea1_50['mean_fitness_std'], 
                fmt='-o', label='EA1 Mean Fitness', color='deepskyblue', markersize=2, linewidth=0.5)
    plt.errorbar(df_ea1_50['gen'], df_ea1_50['best_mean'], yerr=df_ea1_50['best_std'], fmt='--o', 
                label='EA1 Max Fitness', color='lightskyblue', markersize=3, linewidth=0.5)

    # List of EA2 subsections with their respective colors
    ea2_subsections = [
        ('input_to_hidden', 'chocolate', 'navajowhite'),
        ('jump', 'saddlebrown', 'sandybrown'),
        ('release', 'peru', 'linen'),
        ('shoot', 'goldenrod', 'burlywood'),
        ('walk_left', 'sienna', 'peachpuff'),
        ('walk_right', 'darkorange', 'orange'),
    ]
    
    for subsection, color_mean, color_max in ea2_subsections:
        df_ea2 = df[df['algorithm'] == 'EA2' + f'_{subsection}']
        plot_ea2_subsection(df_ea2, enemy_name, save_path, subsection, color_mean, color_max)

    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Fitness', fontsize=16)
    plt.title(f'Fitness Evolution across Generations ({enemy_name})', fontsize=20)
    plt.legend(fontsize=6, loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()


# Set the folders containing the data files for EA1 and EA2
data_folder_ea1 = 'train_ea1_e1'  # Change e[number] to selected enemy
ea2_subfolders = [
    'input_to_hidden',
    'jump',
    'release',
    'shoot',
    'walk_left',
    'walk_right'
]

enemy_name = 'Enemy 1' # Change number to selected enemy
save_path = f'graphs/line-plots/aggregated_fitness_evolution_{enemy_name.lower()}.png'

# Call the function to aggregate data for EA1
df_aggregated_ea1 = aggregate_runs(data_folder_ea1, 'EA1')

# Aggregate data for each EA2 subsection
df_aggregated_ea2_list = []
for subfolder in ea2_subfolders:
    data_folder_ea2 = f'train_ea2_e1/{subfolder}' # Change e[number] to selected enemy
    df_aggregated_ea2 = aggregate_runs(data_folder_ea2, f'EA2_{subfolder}')
    df_aggregated_ea2_list.append(df_aggregated_ea2)

# Combine the aggregated data from EA1 and EA2
df_aggregated = pd.concat([df_aggregated_ea1] + df_aggregated_ea2_list, ignore_index=True)

# Plot the combined data
plot_fitness_evolution(df_aggregated, enemy_name, save_path)

print(f"Aggregated plot saved as {save_path}")