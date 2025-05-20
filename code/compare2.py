import pandas as pd
import os

# Folder containing your CSV files
folder_path = "mc_dropout_results"

# Starting index
start_index = 5613

# Process each CSV in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read CSV
        df = pd.read_csv(file_path)

        # Add ID column
        df.insert(0, 'id', [(f"{i:06d}") for i in range(start_index, start_index + len(df))])

        # Save back (or to a new folder if you prefer)
        df.to_csv(file_path, index=False)

        print(f"Processed: {filename}")
