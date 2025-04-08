import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('predictions_with_outputs.csv')


# Select a specific image
image_name = '007185.jpg'
row = df[df['file_name'] == image_name].iloc[0]

# Extract values for plotting
labels = ['Apparent Age Avg', 'Apparent Age ± Std', 'Real Age', 'Predicted Age ± Std']
values = [
    row['apparent_age_avg'],
    f"{row['apparent_age_avg']} ± {row['apparent_age_std']}",
    row['real_age'],
    f"{row['predicted_age']} ± {row['predicted_std']}"
]

# Plotting numerical values with error bars
x_labels = ['Apparent Age', 'Real Age', 'Predicted Age']
x = list(range(len(x_labels)))
y = [row['apparent_age_avg'], row['real_age'], row['predicted_age']]
yerr = [row['apparent_age_std'], 0, row['predicted_std']]

plt.figure(figsize=(8, 6))
plt.bar(x, y, yerr=yerr, capsize=10, color=['skyblue', 'lightgreen', 'salmon'])
plt.xticks(x, x_labels)
plt.title(f'Age Comparison for Image: {image_name}')
plt.ylabel('Age')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
