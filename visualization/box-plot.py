import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Define the directory where CSV files are located
data_dir = 'testing/'

# Step 2: Define the list of CSV file names and their corresponding labels
csv_files = ['test_ea1_e1.csv', 'test_ea2_e1.csv', 'test_ea1_e7.csv', 
             'test_ea2_e7.csv', 'test_ea1_e8.csv'] # 'test_ea2_e8.csv']
labels = ['EA1-E1', 'EA2-E1', 'EA1-E7', 'EA2-E7', 'EA1-E8']

# Step 3: Create an empty DataFrame to store the gains and their labels
gain_data_combined = pd.DataFrame()

# Step 4: Loop through the CSV files, read each one, and add 'gain' and 'label' columns
for csv_file, label in zip(csv_files, labels):
    file_path = os.path.join(data_dir, csv_file)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Add the 'gain' column and a 'label' column to indicate the source of the data
    df['label'] = label
    gain_data_combined = pd.concat([gain_data_combined, df[['gain', 'label']]])

# Step 5: Prepare the plot
plt.figure(figsize=(12, 6))

# Define a color palette: same color for EA1, another color for EA2
palette = ['#1f77b4', '#ff7f0e']  # EA1: blue, EA2: orange

# Plot the boxplots for the gain data with hue based on the label (EA1 or EA2)
sns.boxplot(x='label', y='gain', data=gain_data_combined, palette=palette[:2])

# Step 6: Set the plot attributes
plt.ylim(-100, 100)
plt.title('Boxplots of Gain for 2 EAs optimized against respective enemy', fontsize=20)
plt.xlabel('EAs optimized to enemy', fontsize=16)
plt.ylabel('Gain', fontsize=16)
plt.xticks(fontsize=12)

# Step 7: Create the directory to save the plot if it doesn't exist
save_dir = 'graphs/box-plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Step 8: Save the plot to the specified folder
plot_path = os.path.join(save_dir, 'gain_boxplot_comparison.png')
plt.savefig(plot_path)

# Step 9: Show the plot
plt.show()

# Confirmation message
print(f"Boxplot saved at {plot_path}")
