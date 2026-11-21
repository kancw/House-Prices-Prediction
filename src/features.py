"""特徵工程（成員 A 負責）。

每個步驟都是一個函式：輸入 DataFrame（train+test 合併、不含 SalePrice），輸出處理後的 DataFrame。
步驟名稱登記在 STEPS，versions.py 用名稱挑選要開啟哪些步驟，方便：
  - 每個版本只多開一個步驟 → 看出每步帶來多少進步
  - ablation：拿掉某一步 → 看分數變差多少（回答「哪個特徵工程最有用」）

新增特徵的方法：寫一個新函式 → 加進 STEPS → 在 versions.py 開一個新版本使用它。
"""
import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET

# NA 代表「沒有這個設施」，不是缺資料（見 data_description.txt）
NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_COLS = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "GarageYrBlt", "GarageCars", "GarageArea", "MasVnrArea",
]
# 品質類欄位有順序：Ex > Gd > TA > Fa > Po > None
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


# ---------------------------------------------------------------- 步驟函式

def fix_types(df):
    """數字形式但其實是類別的欄位 → 轉成字串。"""
    df = df.copy()
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        df[col] = df[col].astype(str)
    return df.drop(columns=["Utilities"])  # 幾乎全部同一個值，沒有資訊


def fill_missing(df):
    """處理缺失值。"""
    df = df.copy()
    df[NONE_COLS] = df[NONE_COLS].fillna("None")
    df[ZERO_COLS] = df[ZERO_COLS].fillna(0)
    # 同一區的房子臨街寬度相近 → 用 Neighborhood 中位數補
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    )
    for col in df.columns:  # 其他少量缺失：類別用眾數、數值用中位數
        if df[col].isna().any():
            is_num = pd.api.types.is_numeric_dtype(df[col])
            fill = df[col].median() if is_num else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
    return df


def encode_ordinal(df):
    """品質欄位轉成 0–5 的分數，保留順序資訊（不做 one-hot）。"""
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP).fillna(0)
    return df


def add_features(df):
    """自己創造的新特徵。"""
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
    """偏態太大的數值欄位做 log1p，讓分佈接近常態（對線性模型特別有幫助）。"""
    df = df.copy()
    numeric = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2]
    skewed = [c for c in numeric if abs(df[c].skew()) > threshold and df[c].min() >= 0]
    df[skewed] = np.log1p(df[skewed])
    return df


# 依執行順序排列；versions.py 用這裡的名稱
STEPS = {
    "fix_types": fix_types,
    "fill_missing": fill_missing,
    "encode_ordinal": encode_ordinal,
    "add_features": add_features,
    "fix_skew": fix_skew,
}


# ---------------------------------------------------------------- 組合

def build_features(train, test, steps, numeric_only=False):
    """依 steps 處理 train+test，回傳 (X_train, X_test)。

    train 和 test 合併處理，確保 one-hot 後欄位一致（這個比賽的常見做法）。
    numeric_only=True：只用數值欄位（baseline 用）。
    """
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"未知的特徵步驟：{unknown}，可用的有 {list(STEPS)}")

    n_train = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True).drop(columns=[ID_COL])

    for name in STEPS:  # 永遠依 STEPS 的固定順序執行
        if name in steps:
            df = STEPS[name](df)

    if numeric_only:
        df = df.select_dtypes("number")
    else:
        df = pd.get_dummies(df, dtype=float)
    df = df.fillna(df.median(numeric_only=True)).fillna(0)  # 保底：模型不能吃 NaN

    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
