import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# Load configuration from the JSON file (single source of truth)
CONFIG_PATH = '/home/ubuntu/tab-ddpm/tab-ddpm-main/data/quiche_highlevel/column_config.json'
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

TARGET_COLUMN = config['target_column']
TO_KEEP = config['to_keep']
CATEGORICAL_COLUMNS = config['categorical_columns']
NUMERICAL_COLUMNS = config['numerical_columns']

CSV_URL = 'https://raw.githubusercontent.com/imets01/synthetic_network_data_gen/refs/heads/anna/high_level_features/dataset/all_captures_dataset.csv'
DATA_DIR = '/home/ubuntu/tab-ddpm/tab-ddpm-main/data/quiche_highlevel'

# Load and Preprocess Data
print(f"Downloading data from {CSV_URL}...")
df = pd.read_csv(CSV_URL, on_bad_lines='skip')
print(f"Original shape: {df.shape}")

df = df.dropna()
print(f"Shape after dropping NaN: {df.shape}")

# Keep only the columns in TO_KEEP + TARGET_COLUMN
columns_to_use = [c for c in TO_KEEP if c in df.columns] + [TARGET_COLUMN]
df = df[columns_to_use]
print(f"Shape after keeping only selected columns: {df.shape}")
print(f"Columns kept: {list(df.columns)}")

# Filter categorical columns to only those in the data
all_categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

# Convert binary categorical columns to 'Yes'/'No'
print("Converting binary categorical columns to 'Yes'/'No'...")
df = df.copy()
for col in all_categorical:
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            print(f"  Converting binary column: {col}")
            df[col] = df[col].map({0: "No", 1: "Yes"}).astype("object")
        else:
            df[col] = df[col].astype("object")
print("✅ Done converting binary categorical columns.")

# Identify Feature Columns
all_cols = list(df.columns)
feature_cols = [c for c in all_cols if c != TARGET_COLUMN]
num_cols = [c for c in feature_cols if c not in all_categorical]

print("\n--- Configuration Complete ---")
print(f"Target Column: {TARGET_COLUMN}")
print(f"Task Type: Regression")
print(f"Numerical Features ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(all_categorical)}): {all_categorical}")
print("---------------------------------\n")

# Split Data
print("Splitting data...")
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=(0.1 / 0.9), random_state=42)
print(f"Train shapes: {train_df.shape}, Val shapes: {val_df.shape}, Test shapes: {test_df.shape}")

def save_split(df_split, split_name):
    print(f"\nProcessing split: {split_name}")
    X_num_df = df_split[num_cols]
    X_cat_df = df_split[all_categorical]
    y_df = df_split[TARGET_COLUMN]

    X_num_np = X_num_df.values.astype(np.float32)
    X_cat_np = X_cat_df.astype(str).values

    task_type = 'regression'
    y_np = y_df.values.astype(np.float32).reshape(-1, 1)

    print(f"  X_num shape: {X_num_np.shape}, X_cat shape: {X_cat_np.shape}, y shape: {y_np.shape}")

    np.save(os.path.join(DATA_DIR, f'X_num_{split_name}.npy'), X_num_np)
    np.save(os.path.join(DATA_DIR, f'X_cat_{split_name}.npy'), X_cat_np)
    np.save(os.path.join(DATA_DIR, f'y_{split_name}.npy'), y_np)
    return task_type

# Save All Splits
task_type = save_split(train_df, 'train')
_ = save_split(val_df, 'val')
_ = save_split(test_df, 'test')

# Create info.json
print("\nCreating info.json...")
info_dict = {
    "name": "Quiche", "id": "quiche--default", "task_type": task_type,
    "n_num_features": len(num_cols), "n_cat_features": len(all_categorical),
    "train_size": train_df.shape[0], "val_size": val_df.shape[0], "test_size": test_df.shape[0],
}

with open(os.path.join(DATA_DIR, 'info.json'), 'w') as f:
    json.dump(info_dict, f, indent=4)

print(f"\n✅ Successfully created dataset in: {DATA_DIR}")
