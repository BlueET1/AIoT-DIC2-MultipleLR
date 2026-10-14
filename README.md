# Regression Explorer：多元線性回歸與特徵選取

互動式網頁，用來觀察多元線性回歸（Multiple Linear Regression）中「**到底需要幾個變數**」這個問題。選一個資料集，比較五種特徵選取方法，看測試誤差在加入第幾個變數後不再下降。

**線上展示：** https://blueet1.github.io/AIoT-DIC2-MultipleLR/

## 功能

- **三種 Problem（右上角下拉選單）**
  - **50 Startups**：用研發、行政、行銷花費與所在州預測新創公司利潤（50 筆）。
  - **Boston Housing**：用 13 個地區特徵預測房價中位數 MEDV（506 筆）。
  - **自訂資料集**：貼上任意 CSV 連結或上傳檔案，直接在瀏覽器內完成分析。
- **五種特徵選取方法**：Lasso、Forward SFS、RFE、ElasticNet、Exhaustive（最佳子集），可單獨查看或「Compare all」疊圖比較。
- **逐步增加變數（k = 1…K）**：每個 k 都顯示測試集 RMSE、R²、Adjusted R²、MAE，以及訓練集的 F 檢定。
- **完整統計輸出**：迴歸方程式、係數表（估計值、標準誤、t、p 值、95% 信賴區間、VIF）、測試集最大殘差。
- **預測計算機**：調整輸入值，即時看目前模型的預測結果。
- 支援深色模式，手機版面可用。

## 使用自訂資料集

選擇下拉選單的 *Custom dataset*，貼上 CSV 連結（例如 [50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)）後按 **Analyze**，或直接上傳檔案。載入後可以改選要預測的目標欄位。

- 需要標題列、至少一個數值欄作為目標，以及至少 20 筆完整資料。
- 文字欄（最多 8 種值）會轉成 0/1 虛擬變數，字母順序第一個值為基準組。
- 有缺值的列會被捨棄；名稱像 `id` 的欄位會被略過；預測變數最多 40 個。
- 預測變數超過 15 個時會略過 Exhaustive（子集數量太多）。
- 連結必須允許跨網域下載（`raw.githubusercontent.com` 可以，GitHub 的 `blob` 連結會自動轉換）。其他網站請先下載再上傳。
- 所有計算都在瀏覽器內執行，資料不會傳到任何伺服器。
- 網址會變成 `?data=…#custom`，可直接分享。

## 方法說明

- **資料切分**：80% 訓練、20% 測試，固定隨機種子。
- **係數檢定**使用訓練集的 OLS；**RMSE、MAE、R²** 使用測試集。
- **Lasso**：依變數在 L1 路徑上進入模型的先後排序。
- **Forward SFS**：每一步加入使 5-fold CV RMSE 最低的變數。
- **RFE**：從全部變數開始，反覆移除標準化係數最小者。
- **ElasticNet**：以交叉驗證選出參數，依係數絕對值排序。
- **Exhaustive**：對每個 k 列舉所有子集，取 CV RMSE 最低者。

> 內建的 50 Startups 結果由 Python 產生；其中 Exhaustive 是以測試集 RMSE 挑選子集，所以 k=3 的 RMSE 會略低於 Lasso 的 k=2。Boston 以及自訂資料集則是用訓練集交叉驗證挑選，避免偷看測試集。

> **關於 Boston Housing：** 變數 B 由各地區的種族組成推導而來，在倫理上有爭議，scikit-learn 1.2 起已移除此資料集。本專案僅作教學比較用途，網頁上也有相同提醒。

## 專案結構

| 檔案 | 說明 |
| --- | --- |
| `index.html` | 網頁主體（HTML、CSS、JavaScript，圖表使用 Chart.js） |
| `regression_engine.js` | 瀏覽器內的分析引擎：CSV 解析、OLS 與統計檢定、五種特徵選取 |
| `model_data.js` / `model_data.json` | 50 Startups 的預先計算結果 |
| `boston_data.js` | Boston Housing 的預先計算結果 |
| `run_boston_feature_selection.py` | 產生 `boston_data.js` 的 Python 腳本 |
| `run_five_feature_selection.py`、`generate_all_plots.py` | 50 Startups 的特徵選取與圖表腳本 |
| `startup_regression_crisp_dm.ipynb` / `.py` | 以 CRISP-DM 流程完成的 50 Startups 分析 |
| `five_feature_selection_50_startups.ipynb` | 五種特徵選取方法的 notebook |
| `50_Startups.csv`、`boston_housing.csv` | 原始資料 |
| `*.png` | 分析過程輸出的圖表 |

## 在本機執行

網頁不需要建置，用任何靜態伺服器開啟即可（直接雙擊 `index.html` 也能看內建的兩個資料集，但自訂資料集的「連結下載」需要透過 http 開啟）：

```bash
python3 -m http.server 8000
# 開啟 http://localhost:8000
```

重新產生預先計算的結果：

```bash
pip install -r requirements.txt
python3 run_boston_feature_selection.py   # 產生 boston_data.js
```

## 資料來源與參考

- 50 Startups：[Avik Jain, 100-Days-Of-ML-Code](https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv)
- Boston Housing：[StatLib, Carnegie Mellon University](http://lib.stat.cmu.edu/datasets/boston)
- Harrison, D. & Rubinfeld, D. (1978). *Hedonic housing prices and the demand for clean air.* Journal of Environmental Economics and Management, 5(1), 81–102.
- 分析流程參考 CRISP-DM（Cross-Industry Standard Process for Data Mining）。
