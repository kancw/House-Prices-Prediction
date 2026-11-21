# House Prices Prediction

Kaggle 比賽：[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
評分：RMSE( log(預測價格), log(真實價格) )，越低越好。

## 專案結構

```
House-Prices-Prediction/
├── data/                       Kaggle 資料（不上傳 git，見 data/README.md）
├── notebooks/
│   ├── 01_eda.ipynb                    A：探索資料、畫圖
│   ├── 02_feature_engineering.ipynb    A：檢查/試新特徵、ablation
│   └── 03_modeling.ipynb               B：跑版本、比模型、調參、畫分數圖
├── src/
│   ├── config.py      路徑與設定（本機 / Kaggle 自動切換）
│   ├── data.py        讀資料、刪 outliers
│   ├── features.py    A：特徵工程步驟
│   ├── models.py      B：模型、CV、blending
│   ├── versions.py    B：實驗版本表（iter1–4 提交 + 只跑 CV 的細分版本）
│   └── pipeline.py    串起全部：跑版本、寫提交檔、記錄分數、ablation
├── submissions/       產生的提交檔 sub_iter1_baseline.csv ...
├── reports/
│   ├── submission_log.csv   提交紀錄表（報告核心）
│   ├── ablation_log.csv     ablation 結果
│   └── figures/             所有圖（報告/簡報用）
├── docs/report_outline.md   報告、影片、個人問答大綱
└── run.py             命令列入口
```

## 分工

| 成員 | 負責 | 主要檔案 |
|---|---|---|
| A：資料與特徵工程 | EDA、缺失值、編碼、新特徵、ablation | `notebooks/01`、`02`、`src/features.py` |
| B：建模與提交 | 版本管理、模型、調參、blending、Kaggle 提交、紀錄表 | `notebooks/03`、`src/models.py`、`src/versions.py` |

## 安裝

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

然後把 Kaggle 資料放進 `data/`。

## 實驗規劃

**主線：4 次 Kaggle 提交**（影片照這 4 步講）

| 提交 | 內容 | 模型 | 證明什麼 |
|---|---|---|---|
| `iter1_baseline` | 原始資料、中位數補值、one-hot、SalePrice 取 log | Ridge | 起點分數 |
| `iter2_cleaned` | + 刪 2 個異常點 + NA 依意義補成 None/0 | Ridge | 資料清理的效果 |
| `iter3_features` | + 序數編碼、新特徵（TotalSF 等）、偏態處理 | Ridge | 特徵工程的效果 |
| `iter4_ensemble` | 同 iter3 | Ridge + Lasso + ElasticNet + GBR 平均 | 換模型/組合的效果 |

**細分：只跑 CV，不提交**（報告用來回答「哪一步最有用」）

| 版本 | 拆開什麼 |
|---|---|
| `iter1_no_log` | iter1 不取 log → log 轉換的效果 |
| `iter2a_outliers_only` / `iter2b_missing_only` | iter2 的兩個改動分開看 |
| `python run.py iter3_features --ablate` | 逐一拿掉 iter3 的每個特徵步驟 |
| `iter4a_lasso` / `iter4b_elasticnet` / `iter4c_gbr` | 單一模型 vs blending |

## 使用

```
python run.py --list                    # 列出所有版本
python run.py iter1_baseline            # → submissions/sub_iter1_baseline.csv + 更新紀錄表
python run.py --all                     # 跑全部（細分版本只跑 CV）
python run.py iter3_features --ablate   # ablation：哪個特徵步驟最有用
```

## 工作流程

1. A 在 `02` notebook 試新特徵 → 有效就寫進 `src/features.py`
2. B 執行 `python run.py iterN_...`
3. B 把 `submissions/sub_iterN_....csv` 上傳到 Kaggle
4. B 把 Kaggle 分數填進 `reports/submission_log.csv` 的 `kaggle_score`
5. commit + push

## 在 Kaggle Notebook 上跑

把整個專案上傳成 Kaggle Dataset（或用 GitHub 連結），在 Notebook 中 `sys.path.append` 到專案資料夾即可，
`config.py` 會自動讀 `/kaggle/input/...` 的資料。
