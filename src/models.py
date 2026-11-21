"""模型與交叉驗證（成員 B 負責）。

評分標準：RMSE( log(預測), log(真實) )，越低越好。
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.config import N_FOLDS, SEED


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, subsample=0.7,
                        colsample_bytree=0.5, random_state=SEED, n_jobs=-1)


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=8,
                         subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
                         random_state=SEED, verbose=-1)


# 名稱 → 建立模型的函式；versions.py 用這裡的名稱。調參就改這裡。
MODELS = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: make_pipeline(RobustScaler(), Ridge(alpha=10)),
    "lasso": lambda: make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=50000)),
    "elasticnet": lambda: make_pipeline(RobustScaler(),
                                        ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=50000)),
    "rf": lambda: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
    "gbr": lambda: GradientBoostingRegressor(n_estimators=3000, learning_rate=0.03, max_depth=4,
                                             max_features="sqrt", min_samples_leaf=15,
                                             loss="huber", random_state=SEED),
    "xgb": _xgb,
    "lgbm": _lgbm,
}


class Blend(BaseEstimator, RegressorMixin):
    """多個模型的加權平均。"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.models]
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.fitted_])
        return np.average(preds, axis=1, weights=self.weights)


def get_model(spec):
    """spec 可以是名稱 "lasso"，或 blend 設定 {"blend": ["lasso", "gbr"], "weights": [...]}"""
    if isinstance(spec, dict):
        return Blend([get_model(name) for name in spec["blend"]], spec.get("weights"))
    if spec not in MODELS:
        raise ValueError(f"未知的模型：{spec}，可用的有 {list(MODELS)}")
    return MODELS[spec]()


# ---------------------------------------------------------------- 訓練與評估

def rmse_log(y_true, y_pred):
    """比賽的評分方式。"""
    y_pred = np.clip(y_pred, 1, None)  # 避免負數無法取 log
    return float(np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2)))


def fit_predict(model, X_train, y, X_pred, log_target):
    """訓練並預測，回傳的是「價格」（不是 log）。"""
    model = clone(model)
    if log_target:
        model.fit(X_train, np.log1p(y))
        return np.expm1(model.predict(X_pred))
    model.fit(X_train, y)
    return model.predict(X_pred)


def cross_validate(model, X, y, log_target):
    """K-fold CV，回傳 (平均分數, 標準差)。y 是原始價格。"""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X):
        pred = fit_predict(model, X.iloc[tr], y.iloc[tr], X.iloc[va], log_target)
        scores.append(rmse_log(y.iloc[va], pred))
    return float(np.mean(scores)), float(np.std(scores))
