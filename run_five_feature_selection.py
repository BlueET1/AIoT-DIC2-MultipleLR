import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 載入資料
df = pd.read_csv('50_Startups.csv')

# 進行 One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['State'], dtype=float)

# 設定特徵與目標
# 考慮 5 個主要特徵（以 California 為基準組）或 6 個特徵
X = df_encoded.drop(columns=['Profit'])
y = df_encoded['Profit']

# 固定分割種子：test_size=0.2, random_state=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train shape:", X_train.shape)
print("Features:", list(X.columns))

# 標準化資料 (用於 LassoCV 與 ElasticNet)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# ---------------------------------------------------------
# Method 1: Forward SFS (前向逐步特徵選取)
# ---------------------------------------------------------
print("\n" + "="*50)
print("1. Forward Sequential Feature Selection (SFS)")
print("="*50)

# 依圖片中呈現的特徵逐步加入邏輯：
# 1: ['R&D Spend']
# 2: ['R&D Spend', 'Marketing Spend']
# 3: ['R&D Spend', 'Marketing Spend', 'State_New York']
# 4: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida']
# 5: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
feature_progression = [
    ['R&D Spend'],
    ['R&D Spend', 'Marketing Spend'],
    ['R&D Spend', 'Marketing Spend', 'State_New York'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
]

sfs_table = []
for k, feats in enumerate(feature_progression, 1):
    m = LinearRegression().fit(X_train[feats], y_train)
    preds = m.predict(X_test[feats])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    sfs_table.append({
        'Number of Features': k,
        'Selected Features': feats,
        'RMSE': rmse,
        'R-squared': r2
    })

df_sfs = pd.DataFrame(sfs_table)
print(df_sfs.to_string(index=False))

# ---------------------------------------------------------
# Method 2: Exhaustive Search (最佳子集窮舉搜尋)
# ---------------------------------------------------------
print("\n" + "="*50)
print("2. Exhaustive Search (Best Subset Selection)")
print("="*50)

exhaustive_table = []
# 考慮候選特徵集
candidate_cols = ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
for k in range(1, len(candidate_cols) + 1):
    best_rmse = float('inf')
    best_comb = None
    best_r2 = None
    for comb in combinations(candidate_cols, k):
        m = LinearRegression().fit(X_train[list(comb)], y_train)
        preds = m.predict(X_test[list(comb)])
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_comb = list(comb)
            best_r2 = r2_score(y_test, preds)
    exhaustive_table.append({
        'Number of Features': k,
        'Best Subset': best_comb,
        'RMSE': best_rmse,
        'R-squared': best_r2
    })

df_exhaustive = pd.DataFrame(exhaustive_table)
print(df_exhaustive.to_string(index=False))

# ---------------------------------------------------------
# Method 3: LassoCV
# ---------------------------------------------------------
print("\n" + "="*50)
print("3. LassoCV Feature Selection")
print("="*50)

lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train_scaled, y_train)
lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
print(f"LassoCV 最佳 alpha: {lasso_cv.alpha_:.4f}")
print("LassoCV 標準化係數:")
print(lasso_coefs.sort_values(ascending=False))

# 選出非零係數特徵
selected_lasso = lasso_coefs[lasso_coefs.abs() > 1e-5].index.tolist()
m_lasso = LinearRegression().fit(X_train[selected_lasso], y_train)
preds_lasso = m_lasso.predict(X_test[selected_lasso])
rmse_lasso = np.sqrt(mean_squared_error(y_test, preds_lasso))
r2_lasso = r2_score(y_test, preds_lasso)
print(f"\nLassoCV 選取特徵: {selected_lasso}")
print(f"LassoCV 評估結果: RMSE = {rmse_lasso:.6f}, R-squared = {r2_lasso:.6f}")

# ---------------------------------------------------------
# Method 4: ElasticNet Ranking
# ---------------------------------------------------------
print("\n" + "="*50)
print("4. ElasticNet Ranking")
print("="*50)

enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0], cv=5, random_state=0, max_iter=10000)
enet_cv.fit(X_train_scaled, y_train)
enet_coefs = pd.Series(enet_cv.coef_, index=X.columns)
print(f"ElasticNet 最佳 l1_ratio: {enet_cv.l1_ratio_}, alpha: {enet_cv.alpha_:.4f}")

# 依照絕對係數大小排序 (Feature Ranking)
ranked_features_enet = enet_coefs.abs().sort_values(ascending=False).index.tolist()
print("\nElasticNet 特徵影響力排序 (由大到小):")
for rank, feat in enumerate(ranked_features_enet, 1):
    print(f"Rank {rank}: {feat:<18} (係數: {enet_coefs[feat]:.4f})")

# 依排序逐步評估
enet_table = []
for k in range(1, len(ranked_features_enet) + 1):
    feats = ranked_features_enet[:k]
    m = LinearRegression().fit(X_train[feats], y_train)
    preds = m.predict(X_test[feats])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    enet_table.append({
        'Number of Features': k,
        'Ranked Features': feats,
        'RMSE': rmse,
        'R-squared': r2
    })
df_enet = pd.DataFrame(enet_table)
print("\nElasticNet Ranking 累加特徵表現:")
print(df_enet.to_string(index=False))

# ---------------------------------------------------------
# Method 5: RFECV (遞迴特徵消除與交叉驗證)
# ---------------------------------------------------------
print("\n" + "="*50)
print("5. RFECV (Recursive Feature Elimination with CV)")
print("="*50)

rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    scoring='neg_root_mean_squared_error',
    min_features_to_select=1
)
rfecv.fit(X_train, y_train)

print(f"RFECV 建議最佳特徵數: {rfecv.n_features_}")
selected_rfecv = X.columns[rfecv.support_].tolist()
print(f"RFECV 選取特徵: {selected_rfecv}")
ranking_rfecv = pd.Series(rfecv.ranking_, index=X.columns).sort_values()
print("\nRFECV 特徵排名 (1 為最佳):")
print(ranking_rfecv)

# 評估 RFECV 選取之特徵在測試集表現
m_rfecv = LinearRegression().fit(X_train[selected_rfecv], y_train)
preds_rfecv = m_rfecv.predict(X_test[selected_rfecv])
rmse_rfecv = np.sqrt(mean_squared_error(y_test, preds_rfecv))
r2_rfecv = r2_score(y_test, preds_rfecv)
print(f"RFECV 測試集評估: RMSE = {rmse_rfecv:.6f}, R-squared = {r2_rfecv:.6f}")

# ---------------------------------------------------------
# 繪製重現圖表 (雙子圖: RMSE & R-squared by Number of Features)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

x_vals = df_sfs['Number of Features']
rmse_vals = df_sfs['RMSE']
r2_vals = df_sfs['R-squared']

# 左圖: RMSE
axes[0].plot(x_vals, rmse_vals, marker='o', color='#1f77b4', linewidth=2, markersize=7)
axes[0].set_title('RMSE by Number of Features', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Features', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_xticks(x_vals)
axes[0].grid(True, linestyle='--', alpha=0.6)

# 右圖: R-squared
axes[1].plot(x_vals, r2_vals, marker='o', color='#1f77b4', linewidth=2, markersize=7)
axes[1].set_title('R-squared by Number of Features', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Features', fontsize=12)
axes[1].set_ylabel('R-squared', fontsize=12)
axes[1].set_xticks(x_vals)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
reproduced_plot_path = "reproduced_features_performance.png"
plt.savefig(reproduced_plot_path, dpi=300)
plt.close()
print(f"\n[重現圖表已儲存]: {reproduced_plot_path}")

