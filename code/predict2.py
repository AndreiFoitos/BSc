import pandas as pd

# Load all 5 CSVs
df1 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\dropconnect100percent_mc_predictions.csv')  # id, y_true
df2 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\flipout100percentlastlayer_mc_predictions.csv')       # id, y_true
df3 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\gt_avg_test.csv')                                 # file_name, apparent_age_avg
df4 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\ensemble100percent_mc_predictions.csv')                                  # id, y_true
df5 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\flipout100percent_mc_predictions.csv')                                  # id, y_true

# Standardize column names
# Keep only relevant columns before renaming and merging
df1 = df1[['id', 'y_true']].rename(columns={'id': 'file_name', 'y_true': 'y_true_1'})
df2 = df2[['id', 'y_true']].rename(columns={'id': 'file_name', 'y_true': 'y_true_2'})
df3 = df3[['file_name', 'apparent_age_avg']].rename(columns={'apparent_age_avg': 'y_true_3'})
df4 = df4[['id', 'y_true']].rename(columns={'id': 'file_name', 'y_true': 'y_true_4'})
df5 = df5[['id', 'y_true']].rename(columns={'id': 'file_name', 'y_true': 'y_true_5'})


# Convert file_name to string in all DataFrames
for df in [df1, df2, df3, df4, df5]:
    df['file_name'] = df['file_name'].astype(str)

# Merge all on 'file_name'
merged = df1.merge(df2, on='file_name')\
            .merge(df3, on='file_name')\
            .merge(df4, on='file_name')\
            .merge(df5, on='file_name')

# Convert all y_true columns to float
for col in ['y_true_1', 'y_true_2', 'y_true_3', 'y_true_4', 'y_true_5']:
    merged[col] = merged[col].astype(float)

# Find mismatches
mismatches = merged[
    (merged['y_true_1'] != merged['y_true_2']) |
    (merged['y_true_1'] != merged['y_true_3']) |
    (merged['y_true_1'] != merged['y_true_4']) |
    (merged['y_true_1'] != merged['y_true_5'])
]

# Output mismatches
print(mismatches)
