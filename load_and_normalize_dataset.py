import glob
import os
import pandas as pd
import numpy as np

# STEP 1: Metadata creation
output_dir = '/mnt/data/eeg_synthetic_dataset'  # ✅ Update if your path differs
file_paths = glob.glob(os.path.join(output_dir, '*.csv'))

data_entries = []
for file in file_paths:
    label = 'alpha' if 'alpha' in file else 'beta'
    data_entries.append({'file': file, 'label': label})

metadata_df = pd.DataFrame(data_entries).sample(frac=1).reset_index(drop=True)

# STEP 2: Data loading and normalization
def load_and_normalize(filepath):
    df = pd.read_csv(filepath, index_col=0)
    data = df.values.T
    return (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

X = []
y = []

label_map = {'alpha': 0, 'beta': 1}
for _, row in metadata_df.iterrows():
    signal = load_and_normalize(row['file'])
    X.append(signal)
    y.append(label_map[row['label']])

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Label shape:", y.shape)
