"""Load data and remove outliers."""
import pandas as pd

from src.config import DATA_DIR, TARGET


def load_data():
    """Return (train, test) DataFrames."""
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Put Kaggle's train.csv / test.csv in data/ first (see README)."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop houses with a huge living area but a low price
    (the 2 points visible in the GrLivArea vs SalePrice scatter plot during EDA)."""
    mask = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
    return train.loc[~mask].reset_index(drop=True)
