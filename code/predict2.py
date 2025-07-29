import pandas as pd
import numpy as np

# Load the CSV files
df1 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\ensemble100percent_mc_predictions.csv')
df2 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\dropconnect100percent_mc_predictions.csv')
df3 = pd.read_csv(r'C:\Users\Andrei\Documents\GitHub\BSc\code\mc_dropout_results\flipout100percent_mc_predictions.csv')

# Combine all DataFrames
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# Group by 'id' and calculate:
# - mean aleatoric uncertainty
# - mean prediction
grouped = df_all.groupby('id').agg({
    'predictive_uncertainty': 'mean',
    'mean_prediction': 'mean'
}).reset_index()

# Compute 95% CI from aleatoric uncertainty (variance)
grouped['aleatoric_ci'] = 1.96 * np.sqrt(grouped['predictive_uncertainty'])

# Merge y_true back in (assuming consistent per id)
df_y_true = df_all[['id', 'y_true']].drop_duplicates(subset='id')
grouped = grouped.merge(df_y_true, on='id', how='left')

# Calculate absolute error
grouped['abs_error'] = np.abs(grouped['mean_prediction'] - grouped['y_true'])

# Apply filters:
# - CI > 0
# - y_true in [0, 100]
# - mean_prediction > 50
filtered = grouped[
    (grouped['aleatoric_ci'] > 0) &
    (grouped['y_true'] >= 0) & 
    (grouped['y_true'] <= 100) &
    (grouped['mean_prediction'] > 50)
]

# Sort by absolute error (largest first) and get top 10
top10_by_error = filtered.sort_values(by='abs_error', ascending=False).head(10)

# Display
print(top10_by_error[['id', 'aleatoric_ci', 'abs_error', 'y_true', 'mean_prediction']])
