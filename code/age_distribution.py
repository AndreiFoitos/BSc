import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv("appa-real-release/appa-real-release/gt_avg_train.csv")  # Replace with your actual filename

# Set style for better aesthetics
sns.set(style="whitegrid")

# Create subplots
plt.figure(figsize=(14, 6))

# Plot apparent age average distribution
plt.subplot(1, 2, 1)
sns.histplot(df['apparent_age_avg'], kde=True, bins=15, color="skyblue")
plt.title("Distribution of Apparent Age Average In Testing Set")
plt.xlabel("Apparent Age (Average)")
plt.ylabel("Count")

# Plot real age distribution
plt.subplot(1, 2, 2)
sns.histplot(df['real_age'], kde=True, bins=15, color="salmon")
plt.title("Distribution of Real Age In Testing Set")
plt.xlabel("Real Age")
plt.ylabel("Count")

# Show plot
plt.tight_layout()
plt.show()
