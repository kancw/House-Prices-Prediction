# Report outline (~10 pages)

Owner and data source in brackets. A = Stages 1 & 2, B = Stages 3 & 4 (see [team_plan.md](team_plan.md)).

## 1. Introduction & problem definition (A, ~1 page)
- Competition goal, data size, metric RMSE(log)
- Four-stage approach overview

## 2. EDA findings & visualizations (A, ~2 pages)
- SalePrice distribution and log transform -> `figures/target_distribution.png`
- Most correlated features -> `figures/top_correlations.png`, `corr_heatmap.png`
- Outliers -> `figures/outliers.png`
- Missing values -> `figures/missing_values.png`

## 3. Data cleaning methodology (A, ~1.5 pages)
- Outlier removal and why
- NA = no facility vs truly missing; LotFrontage by Neighborhood

## 4. Iteration 1 -> 2 score jump (A, ~0.5 page)
- iter1_baseline vs iter2_cleaned
- Breakdown: iter1_no_log, iter2a_outliers_only, iter2b_missing_only

## 5. Feature engineering rationale & ablation (B, ~2 pages)
- Ordinal mapping, new features, interactions, skew handling, and the reasoning
- **Required question: which feature engineering was most useful, and why?** -> `ablation_log.csv`, `figures/ablation.png`

## 6. Model comparison & ensembling (B, ~1.5 pages)
- Single-model CV table (iter4a/4b/4c), tuning, blend
- Score progress chart -> `figures/score_progress.png`

## 7. Best improvement analysis (B, ~0.5 page)
- **Required question: which submission gave the biggest improvement?** -> largest step across iter1-4 in `submission_log.csv`, explained with the breakdown versions

## 8. Conclusion & appendix (B, ~1 page)
- Final score + **screenshot of the Kaggle team name and score**
- What could be improved next
- Notebook / repo links, full submission log

---

# Individual questions (each member writes their own)

1. What is your main role in doing this competition?
2. What were the main difficulties encountered by the whole team? And by you?
3. What have you learned from this competition?

# Video (< 8 minutes, < 600MB, both members on camera)
- A (~4 min): problem overview, EDA charts, cleaning decisions, baseline -> cleaned score
- B (~4 min): feature engineering impact, model comparison, final ensemble score, key takeaways
