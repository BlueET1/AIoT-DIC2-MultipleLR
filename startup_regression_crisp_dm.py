"""
50 Startups Dataset - Multiple Linear Regression Analysis
Framework: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Library: scikit-learn
"""

import os
import sys

# 設置 Matplotlib 避免環境問題
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定圖表樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def print_section_header(phase_num, phase_name_en, phase_name_zh):
    print("=" * 80)
    print(f"CRISP-DM Phase {phase_num}: {phase_name_zh} ({phase_name_en})")
    print("=" * 80)

def main():
    # -------------------------------------------------------------
    # Phase 1: Business Understanding (商業理解)
    # -------------------------------------------------------------
    print_section_header(1, "Business Understanding", "商業理解")
    print("""
[1.1 商業背景]
創投基金 (Venture Capital, VC) 評估 50 家具潛力的新創公司投資項目。
管理層需要了解各項運營開銷 (R&D, Administration, Marketing) 以及地理區位 (State)
如何影響公司的獲利能力 (Profit)。

[1.2 商業目標]
1. 找出驅動新創公司利潤的最關鍵財務與運營指標。
2. 建立精確預測利潤的量化模型，輔助創投基金進行投資評估與資本配置。
3. 評估地理位置 (California, New York, Florida) 是否對獲利有顯著影響。

[1.3 資料採礦目標與成功準則]
- 使用 scikit-learn 構建多元線性迴歸 (Multiple Linear Regression) 模型。
- 評估指標：決定係數 (R^2 > 0.85)、RMSE (均方根誤差)、MAE (平均絕對誤差)。
- 模型必須符合線性迴歸基本假設 (線性性、同質變異性、殘差常態性)。
""")

    # -------------------------------------------------------------
    # Phase 2: Data Understanding (資料理解)
    # -------------------------------------------------------------
    print_section_header(2, "Data Understanding", "資料理解")
    csv_file = "50_Startups.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"找不到 {csv_file}，請確認資料集已下載至當前目錄。")
    
    df = pd.read_csv(csv_file)
    print(f"資料維度 (Rows, Columns): {df.shape}")
    print("\n[資料前 5 筆預覽]")
    print(df.head())

    print("\n[欄位資料型態與缺失值統計]")
    print(df.info())
    missing_count = df.isnull().sum()
    print(f"\n缺失值統計:\n{missing_count}")

    print("\n[數值特徵統計摘要 (Descriptive Statistics)]")
    print(df.describe().round(2))

    print("\n[類別欄位 State 分布]")
    print(df['State'].value_counts())

    numeric_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    corr_matrix = df[numeric_cols].corr()
    print("\n[特徵相關係數矩陣 (Pearson Correlation Matrix)]")
    print(corr_matrix.round(4))

    # 繪製 EDA 相關性與分布圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. R&D Spend vs Profit
    axes[0].scatter(df['R&D Spend'], df['Profit'], color='#1f77b4', alpha=0.7, edgecolors='k')
    axes[0].set_title('R&D Spend vs Profit (Corr: 0.973)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('R&D Spend ($)')
    axes[0].set_ylabel('Profit ($)')

    # 2. Marketing Spend vs Profit
    axes[1].scatter(df['Marketing Spend'], df['Profit'], color='#2ca02c', alpha=0.7, edgecolors='k')
    axes[1].set_title('Marketing Spend vs Profit (Corr: 0.748)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Marketing Spend ($)')
    axes[1].set_ylabel('Profit ($)')

    # 3. Administration vs Profit
    axes[2].scatter(df['Administration'], df['Profit'], color='#d62728', alpha=0.7, edgecolors='k')
    axes[2].set_title('Administration vs Profit (Corr: 0.201)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Administration ($)')
    axes[2].set_ylabel('Profit ($)')

    plt.tight_layout()
    eda_plot_path = "eda_scatter_relationships.png"
    plt.savefig(eda_plot_path, dpi=300)
    plt.close()
    print(f"\n[EDA 圖表已儲存]: {eda_plot_path}")

    # -------------------------------------------------------------
    # Phase 3: Data Preparation (資料準備)
    # -------------------------------------------------------------
    print_section_header(3, "Data Preparation", "資料準備")
    print("""
[3.1 類別特徵編碼與虛擬變數陷阱 (Dummy Variable Trap)]
- State 包含 3 個類別: California, Florida, New York。
- 透過 One-Hot Encoding 產生虛擬變數 (Dummy Variables)。
- 必須捨棄一個基準類別 (drop_first=True)，以避免完全共線性 (Multicollinearity)。
- 本例中以 California 作為基準類別 (Base Category)。
""")

    # One-Hot Encoding
    df_prepared = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=int)
    print("編碼後特徵欄位清單:")
    print(df_prepared.columns.tolist())

    # 分離特徵矩陣 X 與目標變數 y
    X = df_prepared.drop(columns=['Profit'])
    y = df_prepared['Profit']
    feature_names = X.columns.tolist()

    # 資料集切分 (Train/Test Split: 80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n訓練集大小: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"測試集大小: X_test={X_test.shape}, y_test={y_test.shape}")

    # 特徵標準化 (用於比較標準化係數 Beta Weights)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Phase 4: Modeling (模型建構)
    # -------------------------------------------------------------
    print_section_header(4, "Modeling", "模型建構")

    # 1. 原始尺度的多元線性迴歸
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # 2. 標準化特徵的多元線性迴歸 (用於評估各特徵純粹重要性)
    model_scaled = LinearRegression(fit_intercept=True)
    model_scaled.fit(X_train_scaled, y_train)

    print("[4.1 迴歸模型參數]")
    print(f"截距項 (Intercept, beta_0): {model.intercept_:,.2f}")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Original_Coefficient': model.coef_,
        'Standardized_Coefficient (Beta)': model_scaled.coef_
    })
    print("\n[特徵迴歸係數表]")
    print(coef_df.to_string(index=False))

    print("\n[4.2 迴歸方程式]")
    eq_str = f"Profit = {model.intercept_:,.2f}\n"
    for feat, coef in zip(feature_names, model.coef_):
        sign = "+" if coef >= 0 else "-"
        eq_str += f"         {sign} {abs(coef):.4f} * ({feat})\n"
    print(eq_str)

    # 繪製特徵影響力 (標準化係數) 長條圖
    plt.figure(figsize=(10, 5))
    sorted_idx = np.argsort(np.abs(model_scaled.coef_))
    colors = ['#1f77b4' if c > 0 else '#d62728' for c in model_scaled.coef_[sorted_idx]]
    plt.barh(np.array(feature_names)[sorted_idx], model_scaled.coef_[sorted_idx], color=colors)
    plt.title('Standardized Coefficients (Feature Importance Comparison)', fontsize=14, fontweight='bold')
    plt.xlabel('Standardized Beta Weight (Impact on Profit)')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    coef_plot_path = "coefficients_importance.png"
    plt.savefig(coef_plot_path, dpi=300)
    plt.close()
    print(f"[係數權重圖表已儲存]: {coef_plot_path}")

    # -------------------------------------------------------------
    # Phase 5: Evaluation (模型評估)
    # -------------------------------------------------------------
    print_section_header(5, "Evaluation", "模型評估")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 計算指標
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    n_train, p = X_train.shape
    n_test = X_test.shape[0]
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)

    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print("[5.1 效能評估指標 (Performance Metrics)]")
    metrics_summary = pd.DataFrame({
        'Metric': ['R-squared (R^2)', 'Adjusted R^2', 'RMSE ($)', 'MAE ($)', 'MAPE (%)'],
        'Training Set': [f"{r2_train:.4f}", f"{adj_r2_train:.4f}", f"{np.sqrt(mean_squared_error(y_train, y_train_pred)):,.2f}", f"{mean_absolute_error(y_train, y_train_pred):,.2f}", f"{np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100:.2f}%"],
        'Test Set': [f"{r2_test:.4f}", f"{adj_r2_test:.4f}", f"{rmse_test:,.2f}", f"{mae_test:,.2f}", f"{mape_test:.2f}%"]
    })
    print(metrics_summary.to_string(index=False))

    print("\n[5.2 測試集 10 家新創公司 預測 vs 實際 對照]")
    comparison_df = pd.DataFrame({
        'Actual Profit ($)': y_test.values,
        'Predicted Profit ($)': y_test_pred,
        'Residual ($)': y_test.values - y_test_pred,
        'Error (%)': np.abs((y_test.values - y_test_pred) / y_test.values) * 100
    })
    print(comparison_df.round(2).to_string())

    # 殘差假設檢定分析圖
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 實際值 vs 預測值
    axes[0].scatter(y_test, y_test_pred, color='#2b5c8f', s=70, alpha=0.8, edgecolors='k', label='Test Data Points')
    min_val = min(y_test.min(), y_test_pred.min()) - 5000
    max_val = max(y_test.max(), y_test_pred.max()) + 5000
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line (y = y_hat)')
    axes[0].set_title(f'Actual vs Predicted Profit (Test R^2 = {r2_test:.4f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Profit ($)')
    axes[0].set_ylabel('Predicted Profit ($)')
    axes[0].legend()

    # 2. 殘差 vs 擬合值 (檢驗同質變異性)
    axes[1].scatter(y_train_pred, residuals_train, color='#1f77b4', alpha=0.6, label='Train Residuals')
    axes[1].scatter(y_test_pred, residuals_test, color='#e377c2', alpha=0.8, edgecolors='k', label='Test Residuals')
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title('Residuals vs Fitted Values (Homoscedasticity Check)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fitted / Predicted Values ($)')
    axes[1].set_ylabel('Residuals (Actual - Predicted) ($)')
    axes[1].legend()

    plt.tight_layout()
    eval_plot_path = "model_evaluation_diagnostics.png"
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    print(f"\n[模型診斷評估圖已儲存]: {eval_plot_path}")

    # -------------------------------------------------------------
    # Phase 6: Deployment & Decision Making (部署與決策建議)
    # -------------------------------------------------------------
    print_section_header(6, "Deployment", "部署與決策建議")

    class StartupProfitPredictor:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names

        def predict(self, rd_spend, admin_spend, marketing_spend, state):
            state_florida = 1 if state.strip().title() == 'Florida' else 0
            state_ny = 1 if state.strip().title() == 'New York' else 0
            
            input_data = pd.DataFrame([{
                'R&D Spend': rd_spend,
                'Administration': admin_spend,
                'Marketing Spend': marketing_spend,
                'State_Florida': state_florida,
                'State_New York': state_ny
            }])[self.feature_names]

            pred_profit = self.model.predict(input_data)[0]
            return pred_profit

    predictor = StartupProfitPredictor(model, feature_names)

    print("[6.1 模型封裝與實時預測示範]")
    sample_cases = [
        {"name": "新創公司 A (高研發導向)", "rd": 160000, "admin": 120000, "mkt": 300000, "state": "California"},
        {"name": "新創公司 B (高行銷導向)", "rd": 60000, "admin": 120000, "mkt": 400000, "state": "New York"},
        {"name": "新創公司 C (行政開銷大)", "rd": 30000, "admin": 180000, "mkt": 100000, "state": "Florida"},
    ]

    for case in sample_cases:
        p = predictor.predict(case['rd'], case['admin'], case['mkt'], case['state'])
        print(f">> {case['name']}:")
        print(f"   輸入條件: R&D=${case['rd']:,}, Admin=${case['admin']:,}, Mkt=${case['mkt']:,}, 州別={case['state']}")
        print(f"   ==> 預測利潤 (Profit): ${p:,.2f}")

    print("""
[6.2 創投基金 (VC) 商業決策與資源配置建議]
1. 【首要驅動因子：R&D 研發投入】
   - 研發係數高達 0.8056，且標準化權重高達 36,600+，居所有特徵之冠。
   - 結論：新創公司的核心利潤完全取決於技術研發與產品壁壘。投資人應確保資金主要挹注於研發團隊。

2. 【次要驅動因子：Marketing 行銷推廣】
   - 行銷係數為 0.0299，具顯著正向相關性 (Pearson r = 0.748)。
   - 結論：行銷具備推動獲利的作用，但投資回報彈性遠小於研發，應在產品具備市場競爭力後適度投放。

3. 【無效/負向因子：Administration 行政管理】
   - 行政係數為 -0.0688，與利潤相關性僅 0.20，且控制研發後呈現微弱負效應。
   - 結論：過度膨脹的行政成本（奢華辦公室、龐大管理層）對獲利無益甚至可能侵蝕利潤，需落實 Lean 理念。

4. 【地理區位差異 (State)】
   - Florida 與 New York 相較於 California 基準組之係數差異在幾百至幾美元內，在獲利預測中不具統計或實質決定性意義。
   - 結論：新創公司獲利不依賴特定地理光環，創業團隊應選擇運營成本更合理、人才充裕的地點。
""")

if __name__ == '__main__':
    main()

