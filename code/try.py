import pandas as pd

# Load the CSV data
df = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\gt_avg_train.csv')

# Filter rows where real_age is 16
df_16 = df[df['real_age'] == 87]

# Calculate the absolute difference between real age and apparent age average
df_16['age_diff'] = abs(df_16['real_age'] - df_16['apparent_age_avg'])

# Sort by the difference descending and get top 10
top_10_16 = df_16.sort_values(by='age_diff', ascending=False).head(10)

print(top_10_16[['file_name', 'real_age', 'apparent_age_avg', 'age_diff']])
