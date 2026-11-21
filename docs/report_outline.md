# 報告大綱（約 10 頁）

括號內是負責人與資料來源。

## 1. 問題介紹（B，約 1 頁）
- 比賽目標、資料規模、評分方式 RMSE(log)
- 整體方法流程圖

## 2. EDA 發現（A，約 2 頁）
- SalePrice 分佈與 log 轉換 → `figures/target_distribution.png`
- 最相關特徵 → `figures/top_correlations.png`、`corr_heatmap.png`
- Outliers → `figures/outliers.png`
- 缺失值分析 → `figures/missing_values.png`

## 3. 特徵工程（A，約 3 頁）
- 缺失值處理策略（NA = 沒有設施 vs 真的缺）
- 序數編碼、one-hot
- 新特徵及其理由
- 偏態處理

## 4. 模型與提交歷程（B，約 2 頁）
- iter1 → iter4 提交紀錄 → `submission_log.csv`（submitted = True 的列）
- 分數變化圖 → `figures/score_progress.png`
- 每個 iter 內的細分結果（submitted = False 的列）：log 轉換、異常點 vs 缺失值、單一模型 vs blending
- 模型比較、調參

## 5. 指定題
- **哪個特徵工程最有用？為什麼？**（A）→ `iter3_features --ablate` 的 `ablation_log.csv`、`figures/ablation.png`，再加上 iter1_no_log、iter2a/2b 的對照
- **哪次提交進步最大？**（B）→ iter1–4 中 improvement 最大的一次，再用細分版本解釋原因

## 6. 結論（B，約 1 頁）
- 最終分數 + **Kaggle 隊名與分數截圖**
- 可以再改進的方向
- Notebook 連結

---

# 個人問答（每人各自寫，不要互抄）

1. What is your main role in doing this competition?
2. What were the main difficulties encountered by the whole team? And by you?
3. What have you learned from this competition?

# 影片（≤ 8 分鐘、≤ 600MB、兩人都要露臉）
- B：開場介紹（1 分鐘）
- A：EDA + 特徵工程（3.5 分鐘）
- B：模型、提交歷程、結果、結論（3.5 分鐘）
