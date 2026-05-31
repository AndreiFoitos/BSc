import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from matplotlib import gridspec
import pandas as pd
from matplotlib.patches import Patch

# --- Configuration ---
imageAA = '000757'

apparent_gt = 27.3
apparent_age_std = 12.52     # STD (shaded region)
real_age = 16.0

# File paths
image_dir = r'C:\Users\Andrei\Documents\GitHub\BSc\appa-real-release\appa-real-release\train'
image_path = os.path.join(image_dir, f'{imageAA}.jpg')

# --- Plotting setup ---
sns.set_theme(style="whitegrid", font_scale=1.2)
fig = plt.figure(figsize=(11, 6), constrained_layout=True)

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

# --- Add Legend above image ---
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.5, edgecolor='black', label='STD of annotators')
]
ax_img.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.1),  # move higher above the image
    ncol=1,
    fontsize=100,                 # bigger font
    frameon=True,
    prop={'weight':'bold'}
)
# --- Right: Barplot ---
ax_bar = fig.add_subplot(gs[1])

labels = ['Apparent Age', 'Real Age']
ages = [apparent_gt, real_age]

df = pd.DataFrame({'Type': labels, 'Age': ages})
palette = sns.color_palette("Set2")

sns.barplot(
    data=df,
    x='Type',
    y='Age',
    ax=ax_bar,
    palette=palette,
    width=0.6,
    edgecolor='black'
)

# --- Shaded STD region for Apparent Age ---
bar_width = 0.6
ax_bar.add_patch(
    plt.Rectangle(
        (-bar_width / 2, apparent_gt - apparent_age_std),
        bar_width,
        2 * apparent_age_std,
        color='lightblue',
        alpha=0.5,
        label='STD of annotators',
        zorder=2
    )
)

# --- Show the STD value on top of the bar ---
ax_bar.text(
    0,
    apparent_gt + apparent_age_std + 1,
    f'STD: {apparent_age_std:.2f}',
    ha='center',
    fontsize=18,
    fontweight='bold',
    color='black'
)

# --- Value labels ---
for i, age in enumerate(ages):
    ax_bar.text(
        i, age + 1.5,
        f'{age:.1f}',
        ha='center',
        fontsize=25,
        fontweight='bold'
    )

# --- Difference annotation ---
diff = abs(apparent_gt - real_age)
ax_bar.annotate(
    f'Diff: {diff:.1f} yrs',
    xy=(0.5, max(ages) + apparent_age_std + 5),
    ha='center',
    fontsize=20,
    fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3",
              fc='lightgray',
              ec='gray',
              alpha=0.7)
)

# --- Styling ---
ax_bar.set_ylim(0, max(ages) + apparent_age_std + 15)
ax_bar.set_ylabel('Age (years)', fontsize=40, fontweight='bold')
ax_bar.set_xlabel('')

ax_bar.tick_params(axis='x', labelsize=20)
ax_bar.tick_params(axis='y', labelsize=20)
ax_bar.set_xticklabels(labels, fontweight='bold', fontsize=40)

for label in ax_bar.get_yticklabels():
    label.set_fontweight('bold')

# --- Save and Show ---
fig.savefig(f"{imageAA}_gt_real_age_with_std_shaded.pdf",
            format='pdf',
            bbox_inches='tight')
plt.show()
