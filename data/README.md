# data/

把 Kaggle 下載的檔案放在這裡（不會上傳到 git）：

- `train.csv`
- `test.csv`
- `data_description.txt`
- `sample_submission.csv`

下載方式（二選一）：

1. 網頁：比賽頁 → Data → Download All，解壓到這個資料夾
2. Kaggle CLI（需先設定 API token）：
   ```
   kaggle competitions download -c house-prices-advanced-regression-techniques -p data
   ```
   然後解壓 zip。
