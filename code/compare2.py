import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from matplotlib import gridspec
import pandas as pd

# --- Configuration ---
# --- Configuration ---
imageAA = '000757'
apparent_gt = 27.3  # <-- Manually set apparent age here
real_age = 16.0 

# File paths
image_dir = r'C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\train'
image_path = os.path.join(image_dir, f'{imageAA}.jpg')

# --- Plotting setup ---
sns.set_theme(style="whitegrid", font_scale=1.2)
fig = plt.figure(figsize=(11, 6), constrained_layout=True)

# Adjust width ratio: image = 1, graph = 2
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], figure=fig)

# --- Left: Image ---
ax_img = fig.add_subplot(gs[0])
if os.path.exists(image_path):
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title(f'ID: {imageAA}.jpg', fontsize=14, fontweight='bold', pad=12)
else:
    ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center',
                fontsize=12, fontweight='bold')
    ax_img.axis('off')

# --- Right: Barplot ---
ax_bar = fig.add_subplot(gs[1])
labels = ['Apparent Age', 'Real Age']
ages = [apparent_gt, real_age]
df = pd.DataFrame({'Type': labels, 'Age': ages})

palette = sns.color_palette("Set2")

sns.barplot(data=df, x='Type', y='Age', ax=ax_bar, palette=palette, width=0.6, edgecolor='black')

# Add value labels
for i, age in enumerate(ages):
    ax_bar.text(i, age + 1.5, f'{age:.1f}', ha='center',
                fontsize=25, fontweight='bold')

# Annotate age difference
diff = abs(apparent_gt - real_age)
ax_bar.annotate(f'Diff: {diff:.1f} yrs',
                xy=(0.5, max(ages) + 7), xycoords='data',
                ha='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='lightgray', ec='gray', alpha=0.7))

# Styling
ax_bar.set_ylim(0, max(ages) + 15)
ax_bar.set_ylabel('Age (years)', fontsize=40, fontweight='bold')
ax_bar.set_xlabel('')  # 🔥 This line removes the automatic x-axis label ("Type")

# Make tick labels bold and larger
ax_bar.tick_params(axis='x', labelsize=20)
ax_bar.tick_params(axis='y', labelsize=20)
ax_bar.set_xticklabels(labels, fontweight='bold', fontsize=40)
ax_bar.yaxis.set_tick_params(labelsize=20)
for label in ax_bar.get_yticklabels():
    label.set_fontweight('bold')

# --- Save and Show ---
fig.savefig(f"{imageAA}_gt_real_age_seaborn_resized.pdf", format='pdf', bbox_inches='tight')
plt.show()
