"""Boston Housing: 5 種特徵選取方法 × k = 1..13，輸出 boston_data.js 給 index.html 使用。

流程與 50 Startups 相同：train_test_split(test_size=0.2, random_state=0)，
係數檢定 (SE / t / p / CI / VIF / F) 以訓練集 OLS 計算，RMSE / MAE / R² 以測試集計算。
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = 'MEDV'
SEED = 0

df = pd.read_csv('boston_housing.csv')
X = df.drop(columns=[TARGET])
y = df[TARGET]
FEATURES = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler().fit(X_train)
Xs_train = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_rmse(cols):
    """訓練集 5-fold CV RMSE（用於 Forward SFS 與 Exhaustive，避免偷看測試集）。"""
    scores = cross_val_score(LinearRegression(), X_train[list(cols)], y_train,
                             scoring='neg_root_mean_squared_error', cv=kf)
    return -scores.mean()


# ---------- 各方法產生的 nested 特徵序列 ----------
def lasso_ranking():
    alphas, coefs, _ = lasso_path(Xs_train.values, y_train.values, n_alphas=200)
    entry = {}
    for j, f in enumerate(FEATURES):
        nz = np.nonzero(np.abs(coefs[j]) > 1e-10)[0]
        entry[f] = nz[0] if len(nz) else len(alphas)  # alphas 由大到小，越早進入越重要
    cv = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(Xs_train, y_train)
    return sorted(FEATURES, key=lambda f: (entry[f], -abs(cv.coef_[FEATURES.index(f)])))


def forward_sfs_ranking():
    chosen, remaining = [], FEATURES.copy()
    while remaining:
        best = min(remaining, key=lambda f: cv_rmse(chosen + [f]))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def rfe_ranking():
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Xs_train, y_train)
    return [f for _, f in sorted(zip(rfe.ranking_, FEATURES))]


def elasticnet_ranking():
    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0], cv=5,
                        random_state=SEED, max_iter=20000).fit(Xs_train, y_train)
    return [f for f in pd.Series(enet.coef_, index=FEATURES).abs().sort_values(ascending=False).index]


def exhaustive_subsets():
    out = []
    for k in range(1, len(FEATURES) + 1):
        best = min(combinations(FEATURES, k), key=cv_rmse)
        out.append(list(best))
    return out


def nested(ranking):
    return [ranking[:k] for k in range(1, len(ranking) + 1)]


# ---------- 統計量 ----------
def vif_for(cols):
    if len(cols) == 1:
        return {cols[0]: 1.0}
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        r2 = LinearRegression().fit(X_train[others], X_train[c]).score(X_train[others], X_train[c])
        out[c] = float(1 / (1 - r2)) if r2 < 1 else float('inf')
    return out


def evaluate(cols):
    n, p = len(X_train), len(cols)
    A = np.column_stack([np.ones(n), X_train[cols].values])
    beta, *_ = np.linalg.lstsq(A, y_train.values, rcond=None)
    resid = y_train.values - A @ beta
    dof = n - p - 1
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dof)
    tcrit = stats.t.ppf(0.975, dof)
    sst = ((y_train.values - y_train.values.mean()) ** 2).sum()
    f_stat = ((sst - resid @ resid) / p) / sigma2
    f_p = stats.f.sf(f_stat, p, dof)
    vifs = vif_for(cols)

    coefs = [{'feature': 'Intercept (β₀)', 'coef': beta[0], 'stderr': se[0], 'tstat': t[0], 'pvalue': pv[0],
              'ci_lower': beta[0] - tcrit * se[0], 'ci_upper': beta[0] + tcrit * se[0], 'vif': None}]
    for i, c in enumerate(cols, 1):
        coefs.append({'feature': c, 'coef': beta[i], 'stderr': se[i], 'tstat': t[i], 'pvalue': pv[i],
                      'ci_lower': beta[i] - tcrit * se[i], 'ci_upper': beta[i] + tcrit * se[i], 'vif': vifs[c]})

    m = LinearRegression().fit(X_train[cols], y_train)
    pred = m.predict(X_test[cols])
    r2 = r2_score(y_test, pred)
    nt = len(y_test)
    adj = 1 - (1 - r2) * (nt - 1) / (nt - p - 1)
    res = y_test.values - pred
    order = np.argsort(-np.abs(res))[:10]
    outliers = [{'rank': r + 1, 'index': int(y_test.index[i]), 'actual': float(y_test.values[i]),
                 'predicted': float(pred[i]), 'residual': float(res[i]), 'abs_error': float(abs(res[i])),
                 'pct_error': float(abs(res[i]) / y_test.values[i] * 100)} for r, i in enumerate(order)]
    return {'k': p, 'features': cols, 'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
            'mae': float(mean_absolute_error(y_test, pred)), 'r2': float(r2), 'adj_r2': float(adj),
            'f_stat': float(f_stat), 'f_pvalue': float(f_p), 'coefficients': coefs, 'outliers': outliers}


sequences = {
    'Lasso': nested(lasso_ranking()),
    'Forward SFS': nested(forward_sfs_ranking()),
    'RFE': nested(rfe_ranking()),
    'ElasticNet': nested(elasticnet_ranking()),
    'Exhaustive': exhaustive_subsets(),
}
models = {}
for name, seq in sequences.items():
    models[name] = [evaluate(cols) for cols in seq]
    best = min(models[name], key=lambda m: m['rmse'])
    print(f"{name:12s} best k={best['k']:2d} RMSE={best['rmse']:.4f} R2={best['r2']:.4f}  {best['features']}")

# ---------- 資料概況 ----------
DESC = {
    'CRIM': 'Per-capita crime rate by town', 'ZN': 'Share of residential land zoned for lots over 25,000 sq ft (%)',
    'INDUS': 'Share of non-retail business acres (%)', 'CHAS': 'Borders the Charles River (1 = yes)',
    'NOX': 'Nitric oxide concentration (ppm)', 'RM': 'Average rooms per dwelling',
    'AGE': 'Owner-occupied units built before 1940 (%)', 'DIS': 'Weighted distance to 5 Boston employment centres',
    'RAD': 'Index of accessibility to radial highways', 'TAX': 'Property-tax rate per $10,000',
    'PTRATIO': 'Pupil-teacher ratio by town', 'B': '1000(Bk - 0.63)^2, derived from the Black population share',
    'LSTAT': 'Lower-status population (%)',
}
corr = df.corr()[TARGET].drop(TARGET)
overview = [{'feature': f, 'desc': DESC[f], 'mean': float(df[f].mean()), 'std': float(df[f].std()),
             'min': float(df[f].min()), 'median': float(df[f].median()), 'max': float(df[f].max()),
             'corr': float(corr[f])} for f in FEATURES]

payload = {
    'models': models,
    'overview': overview,
    'target': {'name': TARGET, 'mean': float(y.mean()), 'min': float(y.min()), 'max': float(y.max())},
    'meta': {'n': len(df), 'train_n': len(X_train), 'test_n': len(X_test), 'features': len(FEATURES)},
}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


with open('boston_data.js', 'w', encoding='utf-8') as f:
    f.write('window.BOSTON_DATA = ' + json.dumps(clean(payload), ensure_ascii=False, indent=1) + ';\n')
print('wrote boston_data.js')
