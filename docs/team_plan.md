# Team Plan

Two members, four stages. **A owns Stages 1 & 2 (data & cleaning), B owns Stages 3 & 4 (features & modeling).**
Each stage ends in one Kaggle submission.

| Stage | Owner | Version | What changes | Model | Proves |
|---|---|---|---|---|---|
| 1. Baseline | A | `iter1_baseline` | raw data, median fill, one-hot, log(SalePrice) | Ridge | starting score |
| 2. Cleaning | A | `iter2_cleaned` | + remove 2 outliers, fill NA as None/0, LotFrontage by Neighborhood | Ridge | effect of data cleaning |
| 3. Features | B | `iter3_features` | + ordinal quality mapping, new features, skew fix | Ridge | effect of feature engineering |
| 4. Modeling | B | `iter4_ensemble` | same data as Stage 3 | Ridge + Lasso + ElasticNet + GBR blend | effect of better/combined models |

---

## Overview

| Area | A: Stages 1 & 2 (Data & Cleaning) | B: Stages 3 & 4 (Features & Modeling) |
|---|---|---|
| Code ownership | `src/data.py`, cleaning steps in `src/features.py` (`fix_types`, `fill_missing`), `iter1_*` / `iter2*` entries in `src/versions.py`, `notebooks/01_eda.ipynb` | feature steps in `src/features.py` (`encode_ordinal`, `add_features`, `fix_skew`), `src/models.py`, `iter3_*` / `iter4*` entries in `src/versions.py`, `notebooks/02_*`, `notebooks/03_*` |
| Technical tasks | EDA & charts, baseline pipeline, outlier removal, missing-value fixes | domain features, ordinal mapping, ablation, model tuning, blended ensemble |
| Kaggle uploads | `sub_iter1_baseline.csv`, `sub_iter2_cleaned.csv` | `sub_iter3_features.csv`, `sub_iter4_ensemble.csv` + final leaderboard screenshot |
| Report (~10 pages) | Introduction, EDA, data cleaning, iter1 -> iter2 jump (~5 pages) | Feature engineering & ablation, models & ensemble, best improvement, conclusion & appendix (~5 pages) |
| Video (< 8 min) | first ~4 min | last ~4 min |

---

## A: Stages 1 & 2 (Data & Cleaning)

### Setup (shared, A leads)
- [ ] Create the Kaggle team, invite B, agree on a team name
- [ ] Download the data into `data/`, check `python run.py --list` works
- [ ] Make sure B can clone the repo and run the same command

### Stage 1: EDA + baseline
- [ ] Run `notebooks/01_eda.ipynb` on the real data
- [ ] Figures (saved to `reports/figures/`):
  - [ ] `target_distribution.png`: SalePrice skew before/after log
  - [ ] `top_correlations.png`, `corr_heatmap.png`: features most related to price
  - [ ] `outliers.png`: GrLivArea vs SalePrice with the 2 outliers marked
  - [ ] `missing_values.png`: % missing per column
  - [ ] `categorical_vs_price.png`: Neighborhood / OverallQual vs price
- [ ] Write down 3-5 key findings at the bottom of the EDA notebook
- [ ] `python run.py iter1_baseline` -> upload `submissions/sub_iter1_baseline.csv`
- [ ] `python run.py iter1_no_log` (CV only) -> shows how much the log transform helps
- [ ] Fill the Kaggle score into `reports/submission_log.csv`

### Stage 2: data cleaning
- [ ] Review `remove_outliers` in `src/data.py` against the EDA plot
- [ ] Review `fill_missing` in `src/features.py`:
  - [ ] NA -> "None" for "no facility" columns (check each one in `data_description.txt`)
  - [ ] NA -> 0 for the matching area/count columns
  - [ ] LotFrontage filled with the Neighborhood median
- [ ] `python run.py iter2_cleaned` -> upload `submissions/sub_iter2_cleaned.csv`
- [ ] `python run.py iter2a_outliers_only` and `iter2b_missing_only` (CV only) -> which of the two helped more
- [ ] Fill the Kaggle score into the log
- [ ] **Hand off to B** (see Handoffs)

### Report: A's sections (~5 pages)
- [ ] 1. Introduction & problem definition (competition, data size, metric RMSE(log), approach overview) ~1 page
- [ ] 2. EDA findings & visualizations ~2 pages
- [ ] 3. Data cleaning methodology (outliers, None vs truly missing, Neighborhood fill) ~1.5 pages
- [ ] 4. Iteration 1 -> 2 score jump, with the iter1_no_log / iter2a / iter2b breakdown ~0.5 page

### Video: first ~4 minutes
- Problem overview (~1 min)
- EDA charts: skew, outliers, missing values (~1.5 min)
- Cleaning decisions and the baseline -> cleaned score (~1.5 min)

---

## B: Stages 3 & 4 (Features & Modeling)

### Stage 3: feature engineering
- [ ] Review and extend the feature steps in `src/features.py`:
  - [ ] `encode_ordinal`: quality columns Ex/Gd/TA/Fa/Po -> 5..1
  - [ ] `add_features`: TotalSF, TotalBath, HouseAge, RemodAge, TotalPorchSF, Has* flags
  - [ ] Try interaction features (e.g. OverallQual x GrLivArea) in `notebooks/02_feature_engineering.ipynb`; keep only those that improve CV
  - [ ] `fix_skew`: log1p on skewed numeric columns
- [ ] `python run.py iter3_features` -> upload `submissions/sub_iter3_features.csv`
- [ ] `python run.py iter3_features --ablate` -> `reports/ablation_log.csv` + `figures/ablation.png`
- [ ] Fill the Kaggle score into the log

### Stage 4: modeling
- [ ] Compare single models in `notebooks/03_modeling.ipynb` (Ridge, Lasso, ElasticNet, GBR, LightGBM; optionally add HistGradientBoosting to `MODELS` in `src/models.py`)
- [ ] Tune alpha for Ridge / Lasso / ElasticNet and the GBR settings; put the best values in `src/models.py`
- [ ] `python run.py iter4a_lasso`, `iter4b_elasticnet`, `iter4c_gbr` (CV only) -> single model vs blend
- [ ] Decide the blend members (and weights, optional) in `iter4_ensemble`
- [ ] `python run.py iter4_ensemble` -> upload `submissions/sub_iter4_ensemble.csv`
- [ ] Fill the Kaggle score into the log
- [ ] `figures/score_progress.png` from notebook 03
- [ ] **Screenshot of the team name + final leaderboard score**

### Report: B's sections (~5 pages)
- [ ] 5. Feature engineering rationale & ablation study -> answers **"which feature engineering was most useful, and why?"** ~2 pages
- [ ] 6. Model comparison table & ensembling ~1.5 pages
- [ ] 7. Best improvement analysis -> answers **"which submission gave the biggest improvement?"** (uses all four iterations, including A's) ~0.5 page
- [ ] 8. Conclusion, leaderboard screenshot, appendix (notebook links, submission log) ~1 page

### Video: last ~4 minutes
- Feature engineering impact + ablation chart (~1.5 min)
- Model comparison table + final ensemble score (~1.5 min)
- Key takeaways (~1 min)

---

## Handoffs

| When | From -> To | What |
|---|---|---|
| End of Stage 2 | A -> B | `iter2_cleaned` merged on `main`, its CV and Kaggle score in the log, short note on any cleaning decisions B should know |
| End of Stage 4 | B -> A | all four scores in the log, `score_progress.png`, ablation results, screenshot |
| Report draft | both | each writes their own sections, then one person merges and checks page count and figures |

Stage 3 starts from Stage 2's data, so **B can start exploring features in notebook 02 during Stage 2**, but `iter3_features` should only be submitted after `iter2_cleaned` is final.

## Timeline (adjust to the deadline)

| Week | A | B |
|---|---|---|
| 1 | Kaggle team, data, EDA notebook | set up environment, read `data_description.txt`, sketch feature ideas |
| 2 | iter1 submission + iter1_no_log | try features in notebook 02 |
| 3 | iter2 submission + iter2a/2b, hand off | iter3 submission + ablation |
| 4 | start report sections 1-4 | model comparison, tuning |
| 5 | finish report sections 1-4, slides for first half | iter4 submission, screenshot, report sections 5-8, slides for second half |
| 6 | **together**: merge report, record video, individual answers, submit | |

## Shared rules
- Kaggle allows 10 submissions per day per team: only upload the iteration files; tune with CV.
- Only the owner edits their files. `src/versions.py` and `src/features.py` are shared: A edits only the iter1/iter2 entries and the cleaning steps, B only the iter3/iter4 entries and the feature steps.
- Pull before working, commit small, push after each stage.
- Never commit `data/*.csv` (competition rules).
- Each member writes their own individual answers (main role, difficulties, what you learned).

## Final submission checklist
- [ ] Video < 8 min, < 600 MB, both on camera
- [ ] Report ~10 pages with both required questions answered
- [ ] Kaggle team name + final score screenshot
- [ ] Notebook / repo link in the appendix
- [ ] A's individual answers
- [ ] B's individual answers
