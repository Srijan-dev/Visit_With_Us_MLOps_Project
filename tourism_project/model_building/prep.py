
"""Clean the dataset and create reproducible train/test CSV files."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
TARGET = "ProdTaken"

# Use only customer profile information available before the sales interaction.
FEATURES = ['Age', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
POST_CONTACT_COLUMNS = [
    "TypeofContact", "DurationOfPitch", "NumberOfFollowups",
    "ProductPitched", "PitchSatisfactionScore"
]

def prepare_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().copy()

    # Remove source/index identifiers; they do not describe customer behavior.
    data = data.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

    # Standardize a category spelling found in the source data.
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})

    # Exclude sales-interaction fields to keep the model suitable for pre-contact scoring.
    data = data.drop(columns=POST_CONTACT_COLUMNS, errors="ignore")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")
    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    data[TARGET] = data[TARGET].astype(int)

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROJECT_ROOT / "Xtrain.csv", index=False)
    X_test.to_csv(PROJECT_ROOT / "Xtest.csv", index=False)
    y_train.to_frame(name=TARGET).to_csv(PROJECT_ROOT / "ytrain.csv", index=False)
    y_test.to_frame(name=TARGET).to_csv(PROJECT_ROOT / "ytest.csv", index=False)

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing target distribution: {y_test.value_counts().to_dict()}")
    print("Train/test CSV files saved in:", PROJECT_ROOT)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
