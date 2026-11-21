# Report outline (~10 pages)

Owner and data source in brackets.

## 1. Problem introduction (B, ~1 page)
- Competition goal, data size, metric RMSE(log)
- Overall approach diagram

## 2. EDA findings (A, ~2 pages)
- SalePrice distribution and log transform -> `figures/target_distribution.png`
- Most correlated features -> `figures/top_correlations.png`, `corr_heatmap.png`
- Outliers -> `figures/outliers.png`
- Missing values -> `figures/missing_values.png`

## 3. Feature engineering (A, ~3 pages)
- Missing-value strategy (NA = no facility vs truly missing)
- Ordinal encoding, one-hot
- New features and the reasoning behind them
- Skew handling

## 4. Models and submission history (B, ~2 pages)
- iter1 -> iter4 submissions -> `submission_log.csv` (rows with submitted = True)
- Score progress chart -> `figures/score_progress.png`
- Breakdown inside each iter (rows with submitted = False): log transform, outliers vs missing values, single models vs blend
- Model comparison, tuning

## 5. Required questions
- **Which feature engineering was most useful, and why?** (A) -> `ablation_log.csv` and `figures/ablation.png` from `iter3_features --ablate`, plus the iter1_no_log and iter2a/2b controls
- **Which submission gave the biggest improvement?** (B) -> the iter1-4 step with the largest improvement, explained with the breakdown versions

## 6. Conclusion (B, ~1 page)
- Final score + **screenshot of the Kaggle team name and score**
- What could be improved next
- Notebook links

---

# Individual questions (each member writes their own)

1. What is your main role in doing this competition?
2. What were the main difficulties encountered by the whole team? And by you?
3. What have you learned from this competition?

# Video (≤ 8 minutes, ≤ 600MB, both members on camera)
- B: introduction (1 min)
- A: EDA + feature engineering (3.5 min)
- B: models, submission history, results, conclusion (3.5 min)
