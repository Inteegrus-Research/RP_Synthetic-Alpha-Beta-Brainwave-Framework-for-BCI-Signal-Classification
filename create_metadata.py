import glob
import os
import pandas as pd

output_dir = '/mnt/data/eeg_synthetic_dataset'  # ✅ Use your actual path here

# Load metadata from all generated CSVs
file_paths = glob.glob(os.path.join(output_dir, '*.csv'))

# Extract label from filename
data_entries = []
for file in file_paths:
    label = 'alpha' if 'alpha' in file else 'beta'
    data_entries.append({'file': file, 'label': label})

# Convert to DataFrame and shuffle
metadata_df = pd.DataFrame(data_entries)
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Show sample
print(metadata_df.head())
