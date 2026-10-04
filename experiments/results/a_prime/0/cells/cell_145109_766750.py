import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med

Xb, med = prep(m[bcols])

def ridge_pred(X, tr, va, lam):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Z = (X-mu)/sd
    Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[tr].T@y[tr])
    return Zt@w

def ev(X, tr, va, lam=30.0):
    p = ridge_pred(X, tr, va, lam)
    return float(np.mean(np.abs(y[va]-p[va])))

TR13 = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values
# temporal proxy: train 95-403, validate 431
tr_t = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values
va_t = (m.snapshot_day==431).values
# random proxy
rng = np.random.RandomState(0)
va_r = TR13 & (rng.rand(len(y))<0.25); tr_r = TR13 & ~va_r

for lam in [3,10,30,100,300]:
    print(f"lam={lam}: temporal {ev(Xb,tr_t,va_t,lam):.2f}  random {ev(Xb,tr_r,va_r,lam):.2f}")

# feature drift across snapshot days (train rows only)
print("\nDrift of key features by snapshot day (train rows):")
for c in ['spend_84','spend_28','spend_364','spend_all','recency','tenure','x_exp4w','zero_w12','m_spend28']:
    g = m[m.TR13 if False else m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].groupby('snapshot_day')[c].median()
    print(c, g.round(1).values)
# validation rows feature medians (no targets)
mv = base[base.snapshot_day.isin([459,487,515,543])]
print("\nval medians:", {c: round(float(mv[c].median()),1) for c in ['spend_84','spend_28','spend_364','spend_all','recency','tenure','x_exp4w','zero_w12','m_spend28']})