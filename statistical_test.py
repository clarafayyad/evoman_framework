import pandas as pd
from scipy.stats import mannwhitneyu

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
    df = pd.read_csv('testing/' + file_path)

    # Extract EA and enemy group information from the file name
    ea = file_path.split('_')[1]  # Extract EA1 or EA2
    enemy_group = file_path.split('_')[2].replace('.csv', '')  # Extract enemy group (3,7,8) or (4,6,7)

    # Add EA and enemy group columns to the DataFrame
    df['EA'] = ea
    df['Enemy Group'] = enemy_group

    # Append the current DataFrame to the combined DataFrame
    data_combined = pd.concat([data_combined, df], ignore_index=True)

# Perform Mann-Whitney U test for each enemy group
results = {}
alpha = 0.05  # significance level
for enemy_group in ['4,6,7', '3,7,8']:
    # Extract gains for each EA for the current enemy group
    gains_ea1 = data_combined[(data_combined['EA'] == 'ea1') & (data_combined['Enemy Group'] == enemy_group)]['gain']
    gains_ea2 = data_combined[(data_combined['EA'] == 'ea2') & (data_combined['Enemy Group'] == enemy_group)]['gain']

    # Perform the Mann-Whitney U test
    stat, p_value = mannwhitneyu(gains_ea1, gains_ea2, alternative='two-sided')

    # Determine if the result is significant
    significant = p_value < alpha

    # Store the results
    results[(enemy_group)] = (stat, p_value, significant)

# Display and save the results
with open("statistical_test.txt", "w") as f:
    for key, value in results.items():
        group = key
        stat, p_value, significant = value
        significance_text = "Significant difference" if significant else "No significant difference"
        result_string = (f"Mann-Whitney U test for enemy group {group}: "
                         f"U-statistic = {stat}, p-value = {p_value:.4f} - {significance_text}\n")
        print(result_string.strip())  # Print to console
        f.write(result_string)  # Write to file
