df = load_saved('e011_price.parquet')
t = train_targets()
m = t.merge(df, on=['household_key','snapshot_day'], how='left')
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feat if pd.api.types.is_numeric_dtype(m[c])]
cat = [c for c in feat if c not in num]
print('num', len(num), 'cat', len(cat), cat[:25])
X = m[num].astype(float).replace([np.inf,-np.inf], np.nan)
keep = X.notna().mean() > 0.5
num = [c for c in num if keep[c]]; X = X[keep]
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
pref = {}
for i,c in enumerate(num): pref.setdefault(c[:14], []).append(i)
for k, idx in sorted(pref.items(), key=lambda kv: -len(kv[1]))[:28]:
    Zd = np.delete(Z, idx, axis=1)
    md = fit(Zd, y)
    print(f'{k:16s} n={len(idx):3d} drop_mae={md:8.3f} delta={md-base:+7.3f}')