import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Define the directory where CSV files are located
data_dir = '../testing/'

# Step 2: Define the list of CSV file names
csv_files = ['test_ea1_e1.csv', 'test_ea1_e7.csv', 'test_ea1_e8.csv',
             'test_ea2_e1.csv', 'test_ea2_e7.csv', 'test_ea2_e8.csv']

# Create a list of labels for the boxplots
labels = ['EA1-E1', 'EA1-E7', 'EA1-E8', 'EA2-E1', 'EA2-E7', 'EA2-E8']

# Step 3: Create an empty list to store the data
gain_data = []

# Step 4: Loop through the CSV files, read each one, and compute the average 'gain'
for csv_file, label in zip(csv_files, labels):
    file_path = os.path.join(data_dir, csv_file)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Append the 'gain' column and the corresponding label to gain_data
    for gain in df['gain']:
        gain_data.append({'gain': gain, 'label': label})

# Step 5: Convert the gain_data list into a DataFrame
df_combined = pd.DataFrame(gain_data)

# Step 6: Prepare the plot with 6 boxplots side by side
plt.figure(figsize=(12, 6))

# Custom list of colors (replace these with the colors you like)
custom_colors = ['#1f77b4', '#6baed6', '#9ecae1',  # Blue shades
                 '#d95f0e', '#e6550d', '#a63603']  # Orange/brown shades

# Plot the boxplots with different colors using the palette
sns.boxplot(x='label', y='gain', data=df_combined, palette=custom_colors)

# Step 7: Set the plot attributes
plt.ylim(-100, 100)
plt.title('Boxplots of Gain')
plt.xlabel('Algorithm/Enemy')
plt.ylabel('Gain')

# Step 8: Create the directory to save the plot if it doesn't exist
save_dir = '../graphs/box-plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Step 9: Save the plot to the specified folder
plot_path = os.path.join(save_dir, 'gain_boxplot_comparison.png')
plt.savefig(plot_path)

# Step 10: Show the plot
plt.grid(True)
plt.show()

# Confirmation message
print(f"Boxplot saved at {plot_path}")
