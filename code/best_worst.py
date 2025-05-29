import pandas as pd
import matplotlib.pyplot as plt
import os

# Define directories
data_directory = "mc_dropout_results"
output_directory_best = "plots_per_id/best"
output_directory_worst = "plots_per_id/worst"
os.makedirs(output_directory_best, exist_ok=True)
os.makedirs(output_directory_worst, exist_ok=True)

# Model prediction files
files = [
    "dropconnect100percent_mc_predictions.csv",
    "ensemble100percent_mc_predictions.csv",
    "flipout100percent_mc_predictions.csv"
]

# IDs to plot
target_ids = ['006789', '006532', '005684', '006663', '007010',
              '005809', '006332', '006226', '006699', '006516']

# Load predictions into a dictionary
model_results = {}

for file in files:
    path = os.path.join(data_directory, file)
    df = pd.read_csv(path, dtype={'id': str})
    label = file.replace("_mc_predictions.csv", "")
    model_results[label] = df.set_index('id')[['y_true', 'mean_prediction']]

# Generate and save plots
for idx, id_ in enumerate(target_ids):
    plt.figure(figsize=(8, 5))

    y_true = None
    x = []
    y = []

    # Collect predictions for each model
    for label, df in model_results.items():
        if id_ in df.index:
            pred = df.loc[id_, 'mean_prediction']
            y_true = df.loc[id_, 'y_true']
            x.append(label)
            y.append(pred)

    # Plot model predictions as dots
    plt.scatter(x, y, color='blue', label='Mean Prediction', zorder=3)
    
    # Plot ground truth line if available
    if y_true is not None:
        plt.axhline(y=y_true, color='black', linestyle='--', label=f'Ground Truth: {y_true}', zorder=2)

    # Customize plot
    plt.title(f'Predictions for ID {id_}')
    plt.ylabel('Prediction Value')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Determine output directory based on index
    if idx < 5:
        filename = os.path.join(output_directory_best, f"{id_}_predictions.png")
    else:
        filename = os.path.join(output_directory_worst, f"{id_}_predictions.png")

    plt.savefig(filename)
    plt.close()
