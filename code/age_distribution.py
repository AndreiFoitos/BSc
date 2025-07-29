import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\gt_avg_train.csv")

sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
sns.histplot(df['apparent_age_avg'], kde=True, bins=15, color="skyblue")
plt.title("Distribution of Apparent Age Average In Training Set")
plt.xlabel("Apparent Age (Average)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(df['real_age'], kde=True, bins=15, color="salmon")
plt.title("Distribution of Real Age In Training Set")
plt.xlabel("Real Age")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
