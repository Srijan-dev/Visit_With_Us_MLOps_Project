
"""Validate and summarize the tourism customer dataset."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = ['Unnamed: 0', 'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")
    if "ProdTaken" not in data.columns:
        raise ValueError("Target column 'ProdTaken' is required.")
    print("Dataset registration successful.")
    print(f"File: {DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print("Missing values by column:")
    print(data.isna().sum().to_string())
    print("Target distribution:")
    print(data["ProdTaken"].value_counts(dropna=False).sort_index().to_string())
    return data

if __name__ == "__main__":
    main()
