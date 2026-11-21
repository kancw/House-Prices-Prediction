"""實驗版本表。

兩種版本：
  主線（4 個）：提交到 Kaggle，影片照這 4 步講。產生 submissions/sub_<名稱>.csv
  細分（cv_only=True）：只跑 CV、不提交，用來在報告裡回答「哪一步最有用」

另外 `python run.py iter3_features --ablate` 會逐一拿掉 iter3 的特徵步驟，看哪個最重要。

欄位說明
  desc          改了什麼（寫進 submission_log.csv，直接用在報告）
  steps         開啟的特徵工程步驟（見 features.STEPS）
  numeric_only  只用數值欄位
  log_target    SalePrice 先取 log 再訓練
  drop_outliers 刪除 GrLivArea 異常點
  model         模型名稱（見 models.MODELS）或 {"blend": [...]}
  cv_only       True = 只跑 CV，不產生提交檔
"""

_BASE = dict(steps=[], numeric_only=False, log_target=True, drop_outliers=False,
             model="ridge", cv_only=False)

_CLEANED = dict(_BASE, steps=["fix_types", "fill_missing"], drop_outliers=True)

_FEATURES = dict(_CLEANED, steps=_CLEANED["steps"] + ["encode_ordinal", "add_features", "fix_skew"])

VERSIONS = {
    # ---------------- iter1：起點
    "iter1_baseline": dict(
        _BASE, desc="Baseline：原始資料、中位數補值、類別 one-hot、SalePrice 取 log、Ridge"),
    "iter1_no_log": dict(
        _BASE, log_target=False, cv_only=True,
        desc="［對照］同 iter1 但 SalePrice 不取 log → 看 log 轉換帶來多少進步"),

    # ---------------- iter2：資料清理
    "iter2_cleaned": dict(
        _CLEANED, desc="+ 刪除 2 個 GrLivArea 異常點 + NA 依意義補成 None/0"),
    "iter2a_outliers_only": dict(
        _BASE, drop_outliers=True, cv_only=True,
        desc="［細分］iter1 + 只刪異常點"),
    "iter2b_missing_only": dict(
        _BASE, steps=_CLEANED["steps"], cv_only=True,
        desc="［細分］iter1 + 只修正缺失值"),

    # ---------------- iter3：特徵工程（用 --ablate 看每個步驟的貢獻）
    "iter3_features": dict(
        _FEATURES, desc="+ 品質欄位序數編碼、新特徵（TotalSF、TotalBath、HouseAge 等）、偏態 log1p"),

    # ---------------- iter4：模型
    "iter4_ensemble": dict(
        _FEATURES, model={"blend": ["ridge", "lasso", "elasticnet", "gbr"]},
        desc="Blending：Ridge + Lasso + ElasticNet + Gradient Boosting 平均"),
    "iter4a_lasso": dict(_FEATURES, model="lasso", cv_only=True, desc="［細分］iter3 特徵 + 單獨 Lasso"),
    "iter4b_elasticnet": dict(_FEATURES, model="elasticnet", cv_only=True,
                              desc="［細分］iter3 特徵 + 單獨 ElasticNet"),
    "iter4c_gbr": dict(_FEATURES, model="gbr", cv_only=True,
                       desc="［細分］iter3 特徵 + 單獨 Gradient Boosting"),
}

SUBMITTED = [v for v, cfg in VERSIONS.items() if not cfg["cv_only"]]
