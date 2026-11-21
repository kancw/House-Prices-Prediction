"""路徑與全域設定。本機和 Kaggle Notebook 都能直接用。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 在 Kaggle Notebook 上，資料放在 /kaggle/input/... ；本機則放在 data/
_KAGGLE_INPUT = Path("/kaggle/input/house-prices-advanced-regression-techniques")
DATA_DIR = _KAGGLE_INPUT if _KAGGLE_INPUT.exists() else ROOT / "data"

SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SUBMISSION_LOG = REPORT_DIR / "submission_log.csv"
ABLATION_LOG = REPORT_DIR / "ablation_log.csv"

TARGET = "SalePrice"
ID_COL = "Id"

SEED = 42
N_FOLDS = 5
