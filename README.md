# House Prices Prediction

Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
Metric: RMSE( log(predicted price), log(actual price) ), lower is better.

## Project structure

```
House-Prices-Prediction/
├── data/                       Kaggle data (not in git, see data/README.md)
├── notebooks/
│   ├── 01_eda.ipynb                    A: explore the data, make figures
│   ├── 02_feature_engineering.ipynb    A: check/try features, ablation
│   └── 03_modeling.ipynb               B: run versions, compare models, tune, score chart
├── src/
│   ├── config.py      paths and settings (switches automatically between local / Kaggle)
│   ├── data.py        load data, remove outliers
│   ├── features.py    A: feature engineering steps
│   ├── models.py      B: models, CV, blending
│   ├── versions.py    B: experiment versions (iter1-4 submissions + CV-only breakdowns)
│   └── pipeline.py    ties it together: run versions, write submissions, log scores, ablation
├── submissions/       generated submission files sub_iter1_baseline.csv ...
├── reports/
│   ├── submission_log.csv   submission log (core of the report)
│   ├── ablation_log.csv     ablation results
│   └── figures/             all figures (for report/slides)
├── docs/report_outline.md   outline for the report, video and individual questions
└── run.py             command-line entry point
```

## Roles

| Member | Stages | Responsible for | Main files |
|---|---|---|---|
| A | 1 & 2: data & cleaning | EDA, baseline, outliers, missing values; uploads iter1, iter2 | `notebooks/01`, `src/data.py`, cleaning steps in `src/features.py` |
| B | 3 & 4: features & modeling | features, ablation, tuning, ensemble; uploads iter3, iter4 + screenshot | `notebooks/02`, `03`, feature steps in `src/features.py`, `src/models.py` |

Full task list, handoffs and timeline: [docs/team_plan.md](docs/team_plan.md).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then put the Kaggle data in `data/`.

## Experiment plan

**Main line: 4 Kaggle submissions** (the video follows these 4 steps)

| Submission | Data | Model | Shows |
|---|---|---|---|
| `iter1_baseline` | raw data, median fill, one-hot, log(SalePrice) | Ridge | starting score |
| `iter2_cleaned` | + remove 2 outliers + fill NA as None/0 by meaning | Ridge | effect of data cleaning |
| `iter3_features` | + ordinal encoding, new features (TotalSF...), skew fix | Ridge | effect of feature engineering |
| `iter4_ensemble` | same as iter3 | Ridge + Lasso + ElasticNet + GBR average | effect of better/combined models |

**Breakdown: CV only, not submitted** (used in the report to show which step helped most)

| Version | Separates |
|---|---|
| `iter1_no_log` | iter1 without log -> effect of the log transform |
| `iter2a_outliers_only` / `iter2b_missing_only` | the two changes in iter2 |
| `python run.py iter3_features --ablate` | each feature step in iter3, removed one at a time |
| `iter4a_lasso` / `iter4b_elasticnet` / `iter4c_gbr` | single models vs the blend |

## Usage

```
python run.py --list                    # list all versions
python run.py iter1_baseline            # -> submissions/sub_iter1_baseline.csv + update log
python run.py --all                     # run everything (breakdown versions are CV only)
python run.py iter3_features --ablate   # ablation: which feature step helps most
```

## Workflow

1. The stage owner updates their code (A: Stages 1-2, B: Stages 3-4)
2. The owner runs `python run.py iterN_...`
3. The owner uploads `submissions/sub_iterN_....csv` to Kaggle
4. The owner fills the Kaggle score into the `kaggle_score` column of `reports/submission_log.csv`
5. commit + push

## Running on Kaggle Notebook

Upload the project as a Kaggle Dataset (or link the GitHub repo), then `sys.path.append` the project folder
in the notebook. `config.py` reads the data from `/kaggle/input/...` automatically.
