df = load_saved('e011_price.parquet')
t = train_targets()
m = t.merge(df, on=['household_key','snapshot_day'], how='left')
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feat if pd.api.types.is_numeric_dtype(m[c])]
X = m[num].astype(float).replace([np.inf,-np.inf], np.nan)
keep = [c for c in num if X[c].notna().mean() > 0.5]
X = X[keep]
X = X.fillna(X.median())
y = m[TARGET].values.astype(float)
def fit(Z, y, lam=200.0):
    Zb = np.hstack([Z, np.ones((len(Z),1))])
    A = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Zb.T@y)
    return np.abs(Zb@w - y).mean()
mu = X.mean(); sd = X.std().replace(0,1)
Z = ((X-mu)/sd).values
base = fit(Z, y)
print('ridge train MAE (numeric only):', round(base,3))
groups = {}
for i,c in enumerate(keep):
    key = c.split('_')[0][:12]
    groups.setdefault(key, []).append(i)
res = []
for k, idx in groups.items():
    Zd = np.delete(Z, idx, axis=1)
    res.append((fit(Zd, y)-base, k, len(idx)))
for d,k,n in sorted(res, reverse=True)[:25]:
    print(f'{k:14s} n={n:3d} delta={d:+8.3f}')