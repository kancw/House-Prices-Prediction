"""Feature engineering (owned by member A).

Each step is a function: it takes a DataFrame (train+test combined, without SalePrice)
and returns the processed DataFrame. Steps are registered by name in STEPS, and
versions.py picks which steps to enable by name. This makes it easy to:
  - measure how much each step improves the score
  - run an ablation: remove one step and see how much worse the score gets
    (answers "which feature engineering was most useful?")

To add a feature: write a new function -> add it to STEPS -> enable it in versions.py.
"""
import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET

# NA here means "the house has no such facility", not missing data (see data_description.txt)
NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_COLS = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea", "MasVnrArea",
]
# If the house HAS the facility (area > 0) but a descriptive column is NA, the value is
# truly missing (not "None") -> fill with the most common value among houses that have it.
FACILITY_COLS = {
    "GarageArea": ["GarageType", "GarageFinish", "GarageQual", "GarageCond"],
    "TotalBsmtSF": ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
    "MasVnrArea": ["MasVnrType"],
    "PoolArea": ["PoolQC"],
}
# Quality columns are ordered: Ex > Gd > TA > Fa > Po > None
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


# ---------------------------------------------------------------- steps

def fix_types(df):
    """Columns that look numeric but are really categories -> convert to strings."""
    df = df.copy()
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        df[col] = df[col].astype(str)
    return df.drop(columns=["Utilities"])  # almost every row has the same value: no information


def fill_missing(df):
    """Handle missing values."""
    df = df.copy()
    for area_col, cols in FACILITY_COLS.items():
        has = df[area_col].fillna(0) > 0
        for col in cols:
            df.loc[has & df[col].isna(), col] = df.loc[has, col].mode()[0]
    df[NONE_COLS] = df[NONE_COLS].fillna("None")
    df[ZERO_COLS] = df[ZERO_COLS].fillna(0)
    # Garage year: typo in the test set (2207 -> remodel year 2007); no garage -> use the
    # house's build year instead of 0, which would look like a garage built in year 0
    typo = df["GarageYrBlt"] > 2010
    df.loc[typo, "GarageYrBlt"] = df.loc[typo, "YearRemodAdd"]
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    # Houses in the same neighborhood have similar street frontage -> use the neighborhood median
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )
    for col in df.columns:  # remaining few gaps: mode for categories, median for numbers
        if df[col].isna().any():
            is_num = pd.api.types.is_numeric_dtype(df[col])
            fill = df[col].median() if is_num else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
    return df


def encode_ordinal(df):
    """Map quality columns to 0-5 scores to keep their order (instead of one-hot)."""
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP).fillna(0)
    return df


def add_features(df):
    """New hand-crafted features."""
    df = df.copy()
    yr_sold = df["YrSold"].astype(int)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
                          + df["ScreenPorch"] + df["WoodDeckSF"])
    df["HouseAge"] = yr_sold - df["YearBuilt"]
    df["RemodAge"] = yr_sold - df["YearRemodAdd"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    return df


def fix_skew(df, threshold=0.75):
    """Apply log1p to highly skewed numeric columns so they are closer to normal
    (helps linear models most)."""
    df = df.copy()
    numeric = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2]
    skewed = [c for c in numeric if abs(df[c].skew()) > threshold and df[c].min() >= 0]
    df[skewed] = np.log1p(df[skewed])
    return df


# In execution order; versions.py refers to these names
STEPS = {
    "fix_types": fix_types,
    "fill_missing": fill_missing,
    "encode_ordinal": encode_ordinal,
    "add_features": add_features,
    "fix_skew": fix_skew,
}


# ---------------------------------------------------------------- build

def build_features(train, test, steps, numeric_only=False):
    """Apply `steps` to train+test and return (X_train, X_test).

    train and test are processed together so one-hot columns match
    (common practice in this competition).
    numeric_only=True: keep numeric columns only.
    """
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown feature steps: {unknown}. Available: {list(STEPS)}")

    n_train = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True).drop(columns=[ID_COL])

    for name in STEPS:  # always run in the fixed order of STEPS
        if name in steps:
            df = STEPS[name](df)

    if numeric_only:
        df = df.select_dtypes("number")
    else:
        df = pd.get_dummies(df, dtype=float)
    df = df.fillna(df.median(numeric_only=True)).fillna(0)  # safety net: models cannot take NaN

    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
