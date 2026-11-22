"""Experiment versions.

Two kinds:
  Main line (4): submitted to Kaggle; the video follows these 4 steps.
                 Each writes submissions/sub_<name>.csv
  Breakdown (cv_only=True): CV only, not submitted. Used in the report to show
                 which change helped most.

Also, `python run.py iter3_features --ablate` removes each iter3 feature step one at a time.
"""

_BASE = dict(steps=[], numeric_only=False, log_target=True, drop_outliers=False,
             model="ridge", cv_only=False, partial_giant_rule=False)

_CLEANED = dict(_BASE, steps=["fix_types", "fill_missing"], drop_outliers=True)

_FEATURES = dict(
    _CLEANED,
    steps=_CLEANED["steps"] + ["encode_ordinal", "add_features", "add_interactions", "fix_skew"],
)

# iter5: keep only the feature steps the ablation showed are useful
# (encode_ordinal and add_interactions made CV slightly worse)
_FEATURES_PRUNED = dict(_CLEANED, steps=_CLEANED["steps"] + ["add_features", "fix_skew"])

VERSIONS = {
    # ---------------- iter1: starting point
    "iter1_baseline": dict(
        _BASE, desc="Baseline: raw data, median fill, one-hot categories, log(SalePrice), Ridge"),
    "iter1_no_log": dict(
        _BASE, log_target=False, cv_only=True,
        desc="[Control] iter1 without log(SalePrice) -> effect of the log transform"),

    # ---------------- iter2: data cleaning
    "iter2_cleaned": dict(
        _CLEANED,
        desc="+ removed 2 GrLivArea outliers + NA filled by meaning (None/0/mode) + GarageYrBlt fix"),
    "iter2a_outliers_only": dict(
        _BASE, drop_outliers=True, cv_only=True,
        desc="[Breakdown] iter1 + outlier removal only"),
    "iter2b_missing_only": dict(
        _BASE, steps=_CLEANED["steps"], cv_only=True,
        desc="[Breakdown] iter1 + missing-value fix only"),

    # ---------------- iter3: feature engineering (use --ablate to see each step's contribution)
    "iter3_features": dict(
        _FEATURES,
        desc="+ ordinal quality encoding, new features (TotalSF, TotalBath, HouseAge...), interactions, skew log1p"),

    # ---------------- iter4: models
    "iter4_ensemble": dict(
        _FEATURES, model={"blend": ["ridge", "lasso", "elasticnet", "gbr"]},
        desc="Blend: average of Ridge + Lasso + ElasticNet + Gradient Boosting"),
    "iter4a_lasso": dict(_FEATURES, model="lasso", cv_only=True,
                         desc="[Breakdown] iter3 features + Lasso alone"),
    "iter4b_elasticnet": dict(_FEATURES, model="elasticnet", cv_only=True,
                              desc="[Breakdown] iter3 features + ElasticNet alone"),
    "iter4c_gbr": dict(_FEATURES, model="gbr", cv_only=True,
                       desc="[Breakdown] iter3 features + Gradient Boosting alone"),

    # ---------------- iter5: pruned features + new blend
    "iter5_pruned_blend": dict(
        _FEATURES_PRUNED, model={"blend": ["lasso", "gbr", "xgb", "svr"]},
        desc="Pruned features (no ordinal/interactions, per ablation) + blend of Lasso + GBR + XGBoost + SVR"),
    "iter5a_pruned_iter4_blend": dict(
        _FEATURES_PRUNED, model={"blend": ["ridge", "lasso", "elasticnet", "gbr"]}, cv_only=True,
        desc="[Breakdown] pruned features + iter4 blend -> effect of pruning alone"),

    # ---------------- iter6: post-processing for test houses like the 2 outliers
    "iter6_partial_giant": dict(
        _FEATURES_PRUNED, model={"blend": ["lasso", "gbr", "xgb", "svr"]}, partial_giant_rule=True,
        desc="iter5 + price test houses >4000 sq ft sold as Partial by the $/sq ft of the 2 matching train outliers"),
}

SUBMITTED = [v for v, cfg in VERSIONS.items() if not cfg["cv_only"]]