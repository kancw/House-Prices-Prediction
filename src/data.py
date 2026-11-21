"""讀資料、刪 outliers。"""
import pandas as pd

from src.config import DATA_DIR, TARGET


def load_data():
    """回傳 (train, test) 兩個 DataFrame。"""
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"找不到 {train_path}。請先把 Kaggle 的 train.csv / test.csv 放進 data/（見 README）。"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """刪掉「面積很大但價格很低」的異常點（EDA 時在 GrLivArea vs SalePrice 散佈圖上可看到 2 個）。"""
    mask = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
    return train.loc[~mask].reset_index(drop=True)
