import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file paths
file_paths = [
    'test_ea1_3,7,8.csv',
    'test_ea1_4,6,7.csv',
    'test_ea2_3,7,8.csv',
    'test_ea2_4,6,7.csv'
]

# Initialize an empty DataFrame to store the combined data
data_combined = pd.DataFrame()

# Loop through each file, read it, and append to the combined DataFrame
for file_path in file_paths:
    df = pd.read_csv('../testing/' + file_path)

    # Extract EA and enemy group information from the file name
    ea = (file_path.split('_')[1]).upper()  # Extract EA1 or EA2
    enemy_group = file_path.split('_')[2].replace('.csv', '')  # Extract enemy group (3,7,8) or (4,6,7)

    # Add EA and enemy group columns to the DataFrame
    df['EA'] = ea
    df['Enemy Group'] = enemy_group

    # Append the current DataFrame to the combined DataFrame
    data_combined = pd.concat([data_combined, df], ignore_index=True)

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='EA', y='gain', hue='Enemy Group', data=data_combined, hue_order=['4,6,7', '3,7,8'])
plt.title('Boxplot of Gains for EA1 and EA2')
plt.xlabel('Evolutionary Algorithms')
plt.ylabel('Gain')
plt.legend(title='Enemy Groups')
plt.tight_layout()

plt.savefig('../graphs/gain_boxplot_comparison_t2.png')

plt.show()
