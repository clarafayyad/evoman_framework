import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Define the directory where CSV files are located
data_dir = ''

# Step 2: Define the list of CSV file names
csv_files = ['test_ea1_e1.csv', 'test_ea1_e7.csv', 'test_ea1_e8.csv', 
             'test_ea2_e1.csv', 'test_ea2_e7.csv', 'test_ea2_e8.csv']

# Step 3: Create an empty list to store the average gains for each file
gain_data = []

# Step 4: Loop through the CSV files, read each one, and compute the average 'gain'
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the 'gain' column to the gain_data list
    gain_data.append(df['gain'])

# Step 5: Prepare the plot with 6 boxplots side by side
plt.figure(figsize=(12, 6))

# Create a list of labels for the boxplots
labels = ['EA1-E1', 'EA1-E7', 'EA1-E8', 'EA2-E1', 'EA2-E7', 'EA2-E8']

# Plot the boxplots for the gain data from all CSVs
sns.boxplot(data=gain_data)

# Step 6: Set the plot attributes
plt.ylim(-100, 100)
plt.title('Boxplots of Gain for Different CSV Files')
plt.xlabel('Test Files')
plt.ylabel('Gain')
plt.xticks(ticks=range(len(labels)), labels=labels)

# Step 7: Create the directory to save the plot if it doesn't exist
save_dir = 'graphs/box-plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Step 8: Save the plot to the specified folder
plot_path = os.path.join(save_dir, 'gain_boxplot_comparison.png')
plt.savefig(plot_path)

# Step 9: Show the plot
plt.grid(True)
plt.show()

# Confirmation message
print(f"Boxplot saved at {plot_path}")