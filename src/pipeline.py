"""把 data / features / models 串起來：跑一個版本、寫提交檔、記錄分數。
run.py 和 notebooks 都呼叫這裡。"""
from datetime import date

import pandas as pd

from src.config import ABLATION_LOG, ID_COL, SUBMISSION_DIR, SUBMISSION_LOG, TARGET
from src.data import load_data, remove_outliers
from src.features import build_features
from src.models import cross_validate, fit_predict, get_model
from src.versions import VERSIONS


def prepare(cfg, train=None, test=None):
    """依版本設定產生 (X_train, y, X_test, test_ids)。"""
    if train is None:
        train, test = load_data()
    if cfg["drop_outliers"]:
        train = remove_outliers(train)
    X_train, X_test = build_features(train, test, cfg["steps"], cfg["numeric_only"])
    return X_train, train[TARGET], X_test, test[ID_COL]


def evaluate(cfg, data=None):
    """只跑 CV，回傳 (平均, 標準差)。"""
    X_train, y, _, _ = prepare(cfg, *(data or ()))
    return cross_validate(get_model(cfg["model"]), X_train, y, cfg["log_target"])


def run_version(version, submit=True):
    """跑一個版本：CV → 記錄分數；主線版本再用全部 train 訓練 → 寫 submissions/sub_<version>.csv。"""
    cfg = VERSIONS[version]
    submit = submit and not cfg["cv_only"]
    X_train, y, X_test, test_ids = prepare(cfg)
    model = get_model(cfg["model"])

    cv_mean, cv_std = cross_validate(model, X_train, y, cfg["log_target"])
    print(f"[{version}] {cfg['desc']}\n  CV RMSE(log) = {cv_mean:.5f} ± {cv_std:.5f}"
          f"  （特徵數 {X_train.shape[1]}）")

    if submit:
        pred = fit_predict(model, X_train, y, X_test, cfg["log_target"])
        SUBMISSION_DIR.mkdir(exist_ok=True)
        path = SUBMISSION_DIR / f"sub_{version}.csv"
        pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(path, index=False)
        print(f"  提交檔：{path}")
    log_submission(version, cfg["desc"], cv_mean, cv_std, submitted=not cfg["cv_only"])
    return cv_mean, cv_std


def log_submission(version, desc, cv_mean, cv_std, submitted=True):
    """寫入/更新 reports/submission_log.csv。kaggle_score 欄位要在 Kaggle 提交後手動填。"""
    cols = ["version", "submitted", "date", "description", "cv_rmse", "cv_std", "kaggle_score", "note"]
    log = pd.read_csv(SUBMISSION_LOG) if SUBMISSION_LOG.exists() else pd.DataFrame(columns=cols)
    row = {"version": version, "submitted": submitted, "date": date.today().isoformat(),
           "description": desc, "cv_rmse": round(cv_mean, 5), "cv_std": round(cv_std, 5)}
    if version in log["version"].values:
        for k, v in row.items():  # 保留已手動填的 kaggle_score / note
            log.loc[log["version"] == version, k] = v
    else:
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    SUBMISSION_LOG.parent.mkdir(exist_ok=True)
    log[cols].to_csv(SUBMISSION_LOG, index=False)


def ablation(version, steps=None):
    """逐一拿掉 version 的每個特徵步驟，看 CV 分數變差多少（變差越多 = 該步驟越有用）。"""
    cfg = VERSIONS[version]
    data = load_data()
    base, _ = evaluate(cfg, data)
    rows = [{"version": version, "removed_step": "(none)", "cv_rmse": round(base, 5), "delta": 0.0}]
    print(f"[{version}] 完整版 CV = {base:.5f}")
    for step in steps or cfg["steps"]:
        reduced = {**cfg, "steps": [s for s in cfg["steps"] if s != step]}
        score, _ = evaluate(reduced, data)
        rows.append({"version": version, "removed_step": step,
                     "cv_rmse": round(score, 5), "delta": round(score - base, 5)})
        print(f"  拿掉 {step:<15} CV = {score:.5f}  (變化 {score - base:+.5f})")
    result = pd.DataFrame(rows).sort_values("delta", ascending=False)
    ABLATION_LOG.parent.mkdir(exist_ok=True)
    result.to_csv(ABLATION_LOG, index=False)
    return result
