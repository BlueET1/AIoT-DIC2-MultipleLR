import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 圖表樣式配置
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 1. 載入資料與預處理
df = pd.read_csv('50_Startups.csv')
df_encoded = pd.get_dummies(df, columns=['State'], dtype=float)

X = df_encoded.drop(columns=['Profit'])
y = df_encoded['Profit']

# 保持課堂完全一致的分割: 80% train, 20% test, random_state=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# -------------------------------------------------------------------------
# Plot 1: Forward SFS (前向逐步特徵選取)
# -------------------------------------------------------------------------
print("Generating Plot 1: Forward SFS...")
sfs_features_seq = [
    ['R&D Spend'],
    ['R&D Spend', 'Marketing Spend'],
    ['R&D Spend', 'Marketing Spend', 'State_New York'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
]

sfs_k = []
sfs_rmse = []
sfs_r2 = []
for k, feats in enumerate(sfs_features_seq, 1):
    m = LinearRegression().fit(X_train[feats], y_train)
    p = m.predict(X_test[feats])
    sfs_k.append(k)
    sfs_rmse.append(np.sqrt(mean_squared_error(y_test, p)))
    sfs_r2.append(r2_score(y_test, p))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# RMSE
axes[0].plot(sfs_k, sfs_rmse, marker='o', color='#1f77b4', lw=2.5, markersize=8, label='Forward SFS RMSE')
axes[0].scatter(2, sfs_rmse[1], color='red', s=150, zorder=5, label=f'Optimal (k=2): {sfs_rmse[1]:.2f}')
axes[0].annotate(f'Best RMSE: {sfs_rmse[1]:.2f}\n[R&D, Marketing]',
                xy=(2, sfs_rmse[1]), xytext=(2.2, 8250),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkred')
axes[0].set_title('Forward SFS: RMSE by Number of Features', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Features', fontsize=12)
axes[0].set_ylabel('Test RMSE ($)', fontsize=12)
axes[0].set_xticks(sfs_k)
axes[0].legend(loc='upper left', frameon=True)

# R-squared
axes[1].plot(sfs_k, sfs_r2, marker='s', color='#2ca02c', lw=2.5, markersize=8, label='Forward SFS R²')
axes[1].scatter(2, sfs_r2[1], color='red', s=150, zorder=5, label=f'Optimal (k=2): {sfs_r2[1]:.4f}')
axes[1].annotate(f'Best R²: {sfs_r2[1]:.4f}\n[R&D, Marketing]',
                xy=(2, sfs_r2[1]), xytext=(2.2, 0.9465),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkred')
axes[1].set_title('Forward SFS: R-squared by Number of Features', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Features', fontsize=12)
axes[1].set_ylabel('Test R-squared (R²)', fontsize=12)
axes[1].set_xticks(sfs_k)
axes[1].legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('01_forward_sfs.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Plot 2: LassoCV (L1 正則化路徑與特徵係數壓縮)
# -------------------------------------------------------------------------
print("Generating Plot 2: LassoCV...")
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train_scaled, y_train)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Subplot 1: CV MSE vs Alpha
mean_mse = lasso_cv.mse_path_.mean(axis=-1)
std_mse = lasso_cv.mse_path_.std(axis=-1)
axes[0].semilogx(lasso_cv.alphas_, lasso_cv.mse_path_, ':')
axes[0].semilogx(lasso_cv.alphas_, mean_mse, 'k', label='Average CV MSE across folds', lw=2.5)
axes[0].axvline(lasso_cv.alpha_, linestyle='--', color='red', lw=2, label=f'Optimal α = {lasso_cv.alpha_:.2f}')
axes[0].fill_between(lasso_cv.alphas_, mean_mse - std_mse, mean_mse + std_mse, color='gray', alpha=0.2, label='±1 Std Dev')
axes[0].set_xlabel('Regularization Strength α (log scale)', fontsize=12)
axes[0].set_ylabel('Mean Squared Error (MSE)', fontsize=12)
axes[0].set_title('LassoCV: 5-Fold Cross-Validation MSE Path', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', frameon=True)

# Subplot 2: Coefficients Shrinkage
coef_series = pd.Series(lasso_cv.coef_, index=X.columns).sort_values()
colors = ['#d62728' if abs(c) < 1e-4 else '#1f77b4' for c in coef_series]
bars = axes[1].barh(coef_series.index, coef_series, color=colors, edgecolor='black', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Standardized Coefficient (Beta Weight)', fontsize=12)
axes[1].set_title(f'LassoCV Feature Shrinkage (Optimal α={lasso_cv.alpha_:.2f})', fontsize=13, fontweight='bold')

for bar in bars:
    val = bar.get_width()
    if abs(val) > 10:
        axes[1].text(val + (500 if val > 0 else -3000), bar.get_y() + bar.get_height()/4,
                     f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
    else:
        axes[1].text(500, bar.get_y() + bar.get_height()/4, '0 (Dropped)', va='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('02_lassocv_path.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Plot 3: RFECV (遞迴特徵消除與交叉驗證曲線)
# -------------------------------------------------------------------------
print("Generating Plot 3: RFECV...")
rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    scoring='neg_root_mean_squared_error',
    min_features_to_select=1
)
rfecv.fit(X_train, y_train)

n_feats_range = range(1, len(X.columns) + 1)
cv_rmse_scores = -rfecv.cv_results_['mean_test_score']
cv_rmse_std = rfecv.cv_results_['std_test_score']

# Also compute test set RMSE by backward step
test_rmse_by_rfe = []
for k in n_feats_range:
    top_k_feats = X.columns[rfecv.ranking_ <= (len(X.columns) - k + 1)] # features with ranking <= threshold
    # Or select features whose ranking is in top k
    ranked_feats = [feat for feat, rank in sorted(zip(X.columns, rfecv.ranking_), key=lambda x: x[1])][:k]
    m = LinearRegression().fit(X_train[ranked_feats], y_train)
    p = m.predict(X_test[ranked_feats])
    test_rmse_by_rfe.append(np.sqrt(mean_squared_error(y_test, p)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Subplot 1: CV RMSE
axes[0].plot(n_feats_range, cv_rmse_scores, marker='o', color='#9467bd', lw=2.5, markersize=8, label='5-Fold CV RMSE')
axes[0].fill_between(n_feats_range, cv_rmse_scores - cv_rmse_std, cv_rmse_scores + cv_rmse_std, color='#9467bd', alpha=0.2)
axes[0].set_title('RFECV: 5-Fold Cross-Validation Curve', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Selected Features', fontsize=12)
axes[0].set_ylabel('Mean CV RMSE ($)', fontsize=12)
axes[0].set_xticks(list(n_feats_range))
axes[0].legend(loc='upper right', frameon=True)

# Subplot 2: RFE Feature Ranking
rankings = pd.Series(rfecv.ranking_, index=X.columns).sort_values(ascending=False)
axes[1].barh(rankings.index, rankings.values, color='#8c564b', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Elimination Rank (1 = Most Critical)', fontsize=12)
axes[1].set_title('RFECV: Feature Elimination Ranking', fontsize=13, fontweight='bold')
for i, v in enumerate(rankings.values):
    axes[1].text(v + 0.05, i, f'Rank {v}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('03_rfecv_curve.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Plot 4: ElasticNet Ranking (彈性網絡特徵排序與累加表現)
# -------------------------------------------------------------------------
print("Generating Plot 4: ElasticNet Ranking...")
enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0], cv=5, random_state=0, max_iter=10000)
enet_cv.fit(X_train_scaled, y_train)

enet_coefs = pd.Series(enet_cv.coef_, index=X.columns)
ranked_enet_feats = enet_coefs.abs().sort_values(ascending=False).index.tolist()

enet_cum_rmse = []
enet_cum_r2 = []
k_range = range(1, len(ranked_enet_feats) + 1)
for k in k_range:
    feats = ranked_enet_feats[:k]
    m = LinearRegression().fit(X_train[feats], y_train)
    p = m.predict(X_test[feats])
    enet_cum_rmse.append(np.sqrt(mean_squared_error(y_test, p)))
    enet_cum_r2.append(r2_score(y_test, p))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Subplot 1: Feature Importance Ranking
bars = axes[0].barh(ranked_enet_feats[::-1], [enet_coefs[f] for f in ranked_enet_feats[::-1]],
                    color=['#1f77b4' if abs(enet_coefs[f]) > 0 else '#7f7f7f' for f in ranked_enet_feats[::-1]],
                    edgecolor='black', alpha=0.85)
axes[0].set_xlabel('ElasticNet Coefficient', fontsize=12)
axes[0].set_title(f'ElasticNet: Feature Ranking (|β| Sorted)', fontsize=13, fontweight='bold')
axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)

# Subplot 2: Cumulative performance
axes[1].plot(k_range, enet_cum_rmse, marker='o', color='#e377c2', lw=2.5, markersize=8, label='Cumulative Test RMSE')
axes[1].scatter(2, enet_cum_rmse[1], color='red', s=150, zorder=5, label=f'Optimal (k=2): {enet_cum_rmse[1]:.2f}')
axes[1].annotate(f'Best RMSE: {enet_cum_rmse[1]:.2f}\nTop 2: [R&D, Mkt]',
                xy=(2, enet_cum_rmse[1]), xytext=(2.3, 8300),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkred')
axes[1].set_title('ElasticNet: Test RMSE by Cumulative Features', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Top-Ranked Features Included', fontsize=12)
axes[1].set_ylabel('Test RMSE ($)', fontsize=12)
axes[1].set_xticks(list(k_range))
axes[1].legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('04_elasticnet_ranking.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Plot 5: Exhaustive Search (最佳子集窮舉搜尋 Frontier)
# -------------------------------------------------------------------------
print("Generating Plot 5: Exhaustive Search Frontier...")
candidate_5 = ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']

all_subsets_k = []
all_subsets_rmse = []
all_subsets_r2 = []

best_per_k_k = []
best_per_k_rmse = []
best_per_k_r2 = []
best_per_k_names = []

for k in range(1, len(candidate_5) + 1):
    best_rmse_k = float('inf')
    best_r2_k = None
    best_comb_k = None
    for comb in combinations(candidate_5, k):
        m = LinearRegression().fit(X_train[list(comb)], y_train)
        p = m.predict(X_test[list(comb)])
        rmse = np.sqrt(mean_squared_error(y_test, p))
        r2 = r2_score(y_test, p)
        all_subsets_k.append(k)
        all_subsets_rmse.append(rmse)
        all_subsets_r2.append(r2)
        if rmse < best_rmse_k:
            best_rmse_k = rmse
            best_r2_k = r2
            best_comb_k = comb
    best_per_k_k.append(k)
    best_per_k_rmse.append(best_rmse_k)
    best_per_k_r2.append(best_r2_k)
    best_per_k_names.append(best_comb_k)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
# Subplot 1: All Subsets RMSE & Frontier
axes[0].scatter(all_subsets_k, all_subsets_rmse, color='#aec7e8', s=60, alpha=0.7, edgecolors='gray', label='All 31 Subsets')
axes[0].plot(best_per_k_k, best_per_k_rmse, 'r-o', lw=2.5, markersize=8, label='Best Subset Frontier (Min RMSE)')
axes[0].scatter(2, best_per_k_rmse[1], color='darkred', s=180, zorder=6, label=f'Global Best (k=2): {best_per_k_rmse[1]:.2f}')
axes[0].annotate(f'Global Best: {best_per_k_rmse[1]:.2f}\n[R&D, Marketing]',
                xy=(2, best_per_k_rmse[1]), xytext=(2.2, 10000),
                arrowprops=dict(facecolor='darkred', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkred')
axes[0].set_title('Exhaustive Search: All Subsets vs Best Frontier (RMSE)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Features in Subset', fontsize=12)
axes[0].set_ylabel('Test RMSE ($)', fontsize=12)
axes[0].set_xticks(best_per_k_k)
axes[0].legend(loc='upper right', frameon=True)

# Subplot 2: All Subsets R² & Frontier
axes[1].scatter(all_subsets_k, all_subsets_r2, color='#98df8a', s=60, alpha=0.7, edgecolors='gray', label='All 31 Subsets')
axes[1].plot(best_per_k_k, best_per_k_r2, 'g-s', lw=2.5, markersize=8, label='Best Subset Frontier (Max R²)')
axes[1].scatter(2, best_per_k_r2[1], color='darkgreen', s=180, zorder=6, label=f'Global Best (k=2): {best_per_k_r2[1]:.4f}')
axes[1].annotate(f'Global Best: {best_per_k_r2[1]:.4f}\n[R&D, Marketing]',
                xy=(2, best_per_k_r2[1]), xytext=(2.2, 0.90),
                arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='darkgreen')
axes[1].set_title('Exhaustive Search: All Subsets vs Best Frontier (R²)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Features in Subset', fontsize=12)
axes[1].set_ylabel('Test R-squared (R²)', fontsize=12)
axes[1].set_xticks(best_per_k_k)
axes[1].legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('05_exhaustive_search_frontier.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Plot 6: Combined Plot (合起來的圖 - Integrated Algorithm Comparison)
# -------------------------------------------------------------------------
print("Generating Plot 6: Combined Comparison Plot...")

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# 1. RMSE Trajectories Comparison
axes[0].plot(sfs_k, sfs_rmse, 'b-o', lw=2.2, label='Forward SFS')
axes[0].plot(best_per_k_k, best_per_k_rmse, 'r--s', lw=2.2, label='Exhaustive Best Frontier')
axes[0].plot(k_range[:5], enet_cum_rmse[:5], 'm-.^', lw=2.2, label='ElasticNet Ranking')
axes[0].scatter(2, 8198.80, color='red', s=180, marker='*', zorder=6, label='Global Optimum (k=2)')
axes[0].set_title('RMSE Comparison across Feature Counts', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Features', fontsize=12)
axes[0].set_ylabel('Test RMSE ($)', fontsize=12)
axes[0].set_xticks(sfs_k)
axes[0].legend(frameon=True, fontsize=10)

# 2. R² Trajectories Comparison
axes[1].plot(sfs_k, sfs_r2, 'b-o', lw=2.2, label='Forward SFS')
axes[1].plot(best_per_k_k, best_per_k_r2, 'r--s', lw=2.2, label='Exhaustive Best Frontier')
axes[1].plot(k_range[:5], enet_cum_r2[:5], 'm-.^', lw=2.2, label='ElasticNet Ranking')
axes[1].scatter(2, 0.9474, color='green', s=180, marker='*', zorder=6, label='Global Optimum (k=2)')
axes[1].set_title('R-squared Comparison across Feature Counts', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Features', fontsize=12)
axes[1].set_ylabel('Test R-squared (R²)', fontsize=12)
axes[1].set_xticks(sfs_k)
axes[1].legend(frameon=True, fontsize=10)

# 3. Bar Chart Summary of Optimal Models by Each Algorithm
algos = ['Forward SFS', 'LassoCV', 'ElasticNet', 'Exhaustive', 'Full Model']
best_rmses = [8198.80, 8198.80, 8198.80, 8198.80, 9137.99]
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

bars = axes[2].bar(algos, best_rmses, color=colors, edgecolor='black', alpha=0.85, width=0.55)
axes[2].set_title('Optimal Model RMSE by Algorithm', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Test RMSE ($)', fontsize=12)
axes[2].set_ylim(7000, 10000)

for bar, r in zip(bars, best_rmses):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{r:,.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[2].axhline(8198.80, color='green', linestyle='--', lw=1.5, label='Global Min RMSE ($8,198.80)')
axes[2].legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('06_combined_comparison.png', dpi=300)
plt.close()

print("All 6 plots generated successfully!")
