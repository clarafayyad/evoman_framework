import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'gen': [0, 1, 2, 3, 4],
    'best': [87.320213, 90.859688, 91.971655, 91.971655, 91.971655],
    'mean': [3.473433, 12.042275, 23.725689, 42.1981, 48.396121],
    'std': [22.660653, 30.767312, 35.7294, 39.858457, 40.5681]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Creating a line-plot across generations using average and standard deviation (std)
# def plot_fitness_evolution(file_path, enemy_name):
    # data = pd.read_csv(file_path)

def plot_fitness_evolution(df, enemy_name, save_path):
    # Calculate average mean fitness and std across the generations
    generations = data['gen']
    avg_mean_fitness = data['mean']
    avg_std_mean = data['std']
    
    # Calculate average maximum fitness and std
    max_fitness = data['best']

    # Plotting the mean fitness evolution
    plt.figure(figsize=(10, 6))

    plt.errorbar(generations, avg_mean_fitness, yerr=avg_std_mean, fmt='-o', label='Mean Fitness', color='blue')
    plt.plot(generations, max_fitness, label='Max Fitness', color='red')

    # Labels and title
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Fitness', fontsize=16)
    plt.title(f'Fitness Evolution across Generations (Enemy: {enemy_name})', fontsize=20)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True)
    
    # Create save file
    plt.savefig(save_path)

    # Plot Graph
    plt.show()

# Set the file path where the image will be saved
enemy_name = 'Enemy_1'
save_path = f'results/line-plots/aggregated_fitness_evolution_{enemy_name.lower()}.png'

# Call the function to plot and save the graph
plot_fitness_evolution(df, enemy_name, save_path)

print(f"Plot saved as {save_path}")

