import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Define the directory where CSV files are located
data_dir = 'testing/'

# Step 2: Define the list of CSV file names
csv_files = ['test_ea1_e1.csv', 'test_ea2_e1.csv', 'test_ea1_e7.csv', 
             'test_ea2_e7.csv', 'test_ea1_e8.csv'] # 'test_ea2_e8.csv']

# Step 3: Create an empty list to store the gains for each file
gain_data = []

# Create lists to store the individual EA and enemy labels
ea_labels = []
enemy_labels = []

# Step 4: Loop through the CSV files, read each one, and compute the gains
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the 'gain' column to the gain_data list
    gain_data.append(df['gain'])
    
    # Append EA labels based on the file name (EA1 or EA2)
    if 'ea1' in csv_file:
        ea_labels.append('EA1')
    else:
        ea_labels.append('EA2')
    
    # Append enemy labels based on the enemy number in the file name (E1, E7, E8)
    if 'e1' in csv_file:
        enemy_labels.append('E1')
    elif 'e7' in csv_file:
        enemy_labels.append('E7')
    elif 'e8' in csv_file:
        enemy_labels.append('E8')

# Step 5: Prepare the plot
plt.figure(figsize=(12, 6))

# Combine the gain data and labels into a DataFrame for Seaborn
gain_df = pd.DataFrame({
    'Gain': pd.concat(gain_data, ignore_index=True),
    'EA': [label for label in ea_labels for _ in range(len(gain_data[0]))],  # Expand EA labels
    'Enemy': [label for label in enemy_labels for _ in range(len(gain_data[0]))]  # Expand enemy labels
})

# Step 6: Plot the boxplots, grouped by the enemy and split by EA
sns.boxplot(x='Enemy', y='Gain', hue='EA', data=gain_df)

# Step 7: Set the plot attributes
plt.ylim(-100, 100)
plt.title('Boxplots of Gain for EAs optimized against respective enemies', fontsize=20)
plt.xlabel('Enemy', fontsize=16)
plt.ylabel('Gain', fontsize=16)
plt.legend(title='EA Type', fontsize=12)

# Step 8: Create the directory to save the plot if it doesn't exist
save_dir = 'graphs/box-plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Step 9: Save the plot to the specified folder
plot_path = os.path.join(save_dir, 'gain_boxplot_comparisonv2.png')
plt.savefig(plot_path)

# Step 10: Show the plot
plt.show()

# Confirmation message
print(f"Boxplot saved at {plot_path}")
