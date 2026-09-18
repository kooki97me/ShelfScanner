import pandas as pd
from pathlib import Path

# Find the CSV file inside the data folder
data_folder = Path("data")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("❌ No CSV file found inside the data folder.")
    exit()

file_path = csv_files[0]

print(f"\n📚 Loading: {file_path}")

# Read the dataset
df = pd.read_csv(file_path)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)