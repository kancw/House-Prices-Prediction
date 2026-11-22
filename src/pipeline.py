"""Glue for data / features / models: run a version, write the submission, log the score.
Used by run.py and the notebooks.

Fair CV: features are always built for all 1,460 training houses and the 5 folds are
always the same. Outlier removal only affects which houses a model is TRAINED on; every
version is scored on the same validation houses, so CV scores are comparable.
"""
from datetime import date

import numpy as np
import pandas as pd

from src.config import ABLATION_LOG, ID_COL, SUBMISSION_DIR, SUBMISSION_LOG, TARGET
from src.data import load_data, outlier_mask
from src.features import build_features
from src.models import cv_predict, fit_predict, get_model, rmse_log
from src.versions import VERSIONS


def prepare(cfg, train=None, test=None):
    """Build (X_train, y, X_test, test_ids, train_mask) for a version config.
    X_train / y cover all training houses; train_mask is False for houses the model
    should not be trained on (the outliers, if the version drops them)."""
    if train is None:
        train, test = load_data()
    X_train, X_test = build_features(train, test, cfg["steps"], cfg["numeric_only"])
    is_outlier = outlier_mask(train).to_numpy()
    train_mask = ~is_outlier if cfg["drop_outliers"] else np.ones(len(train), bool)
    return X_train, train[TARGET], X_test, test[ID_COL], train_mask


def score_cv(cfg, data=None):
    """Fair CV for a version. Returns a dict with
      cv_rmse / cv_std: scored on all houses (main number, comparable across versions)
      cv_rmse_normal:   scored on all houses except the 2 outliers
      oof:              out-of-fold predictions for every training house
    """
    data = data or load_data()
    train = data[0]
    X, y, _, _, train_mask = prepare(cfg, *data)
    folds, oof = cv_predict(get_model(cfg["model"]), X, y, cfg["log_target"], train_mask)
    normal = ~outlier_mask(train).to_numpy()
    all_scores = [rmse_log(y.iloc[va], p) for va, p in folds]
    normal_scores = [rmse_log(y.iloc[va][normal[va]], p[normal[va]]) for va, p in folds]
    return {"cv_rmse": float(np.mean(all_scores)), "cv_std": float(np.std(all_scores)),
            "cv_rmse_normal": float(np.mean(normal_scores)), "fold_scores": all_scores,
            "oof": oof}


def evaluate(cfg, data=None):
    """CV only. Returns (mean, std) scored on all houses."""
    s = score_cv(cfg, data)
    return s["cv_rmse"], s["cv_std"]


def run_version(version, submit=True):
    """Run a version: fair CV -> log the score. Main-line versions are also refit on the
    training houses and written to submissions/sub_<version>.csv."""
    cfg = VERSIONS[version]
    submit = submit and not cfg["cv_only"]
    data = load_data()
    s = score_cv(cfg, data)
    print(f"[{version}] {cfg['desc']}\n"
          f"  CV RMSE(log) = {s['cv_rmse']:.5f} ± {s['cv_std']:.5f}"
          f"  | normal houses only = {s['cv_rmse_normal']:.5f}")

    if submit:
        X_train, y, X_test, test_ids, train_mask = prepare(cfg, *data)
        model = get_model(cfg["model"])
        pred = fit_predict(model, X_train[train_mask], y[train_mask], X_test, cfg["log_target"])
        SUBMISSION_DIR.mkdir(exist_ok=True)
        path = SUBMISSION_DIR / f"sub_{version}.csv"
        pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(path, index=False)
        print(f"  Submission file: {path}  ({X_train.shape[1]} features)")
    log_submission(version, cfg["desc"], s, submitted=not cfg["cv_only"])
    return s["cv_rmse"], s["cv_std"]


LOG_COLS = ["version", "submitted", "date", "description",
            "cv_rmse", "cv_std", "cv_rmse_normal", "kaggle_score", "note"]


def log_submission(version, desc, scores, submitted=True):
    """Add/update a row in reports/submission_log.csv.
    Fill in kaggle_score by hand after submitting on Kaggle."""
    log = (pd.read_csv(SUBMISSION_LOG) if SUBMISSION_LOG.exists()
           else pd.DataFrame(columns=LOG_COLS)).reindex(columns=LOG_COLS)
    row = {"version": version, "submitted": submitted, "date": date.today().isoformat(),
           "description": desc, "cv_rmse": round(scores["cv_rmse"], 5),
           "cv_std": round(scores["cv_std"], 5),
           "cv_rmse_normal": round(scores["cv_rmse_normal"], 5)}
    if version in log["version"].values:
        for k, v in row.items():  # keep kaggle_score / note that were filled in by hand
            log.loc[log["version"] == version, k] = v
    else:
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    SUBMISSION_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(SUBMISSION_LOG, index=False)


def ablation(version, steps=None):
    """Remove each feature step of `version` one at a time and see how much the CV score
    gets worse (bigger increase = more useful step)."""
    cfg = VERSIONS[version]
    data = load_data()
    base, _ = evaluate(cfg, data)
    rows = [{"version": version, "removed_step": "(none)", "cv_rmse": round(base, 5), "delta": 0.0}]
    print(f"[{version}] full CV = {base:.5f}")
    for step in steps or cfg["steps"]:
        reduced = {**cfg, "steps": [s for s in cfg["steps"] if s != step]}
        score, _ = evaluate(reduced, data)
        rows.append({"version": version, "removed_step": step,
                     "cv_rmse": round(score, 5), "delta": round(score - base, 5)})
        print(f"  without {step:<15} CV = {score:.5f}  (change {score - base:+.5f})")
    result = pd.DataFrame(rows).sort_values("delta", ascending=False)
    ABLATION_LOG.parent.mkdir(exist_ok=True)
    result.to_csv(ABLATION_LOG, index=False)
    return result
