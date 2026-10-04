import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
st = load_saved('structure_v1.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left').merge(st, on=['household_key','snapshot_day'], how='left', suffixes=('','_s'))
y = m.future_spend_4w.values.astype(float)

def prep(df):
    X = df.copy()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(col.median() if col.notna().any() else 0)
    return X.values

bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
scols = [c for c in st.columns if c not in ('household_key','snapshot_day')]
Xb = prep(m[bcols]); Xs = prep(m[scols])
Xall = np.hstack([Xb, Xs])

tr = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values
rng = np.random.RandomState(0)
def ridge_mae(X, mask_tr, mask_va, lam=30.0):
    mu, sd = X[mask_tr].mean(0), X[mask_tr].std(0)+1e-9
    Z = (X-mu)/sd
    Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[mask_tr].T@Zt[mask_tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[mask_tr].T@y[mask_tr])
    return float(np.mean(np.abs(y[mask_va] - Zt[mask_va]@w)))

va = tr & (rng.rand(len(y))<0.25)
trp = tr & ~va
print("ridge base MAE:", round(ridge_mae(Xb, trp, va),3))
print("ridge base+structure MAE:", round(ridge_mae(Xall, trp, va),3))

mu, sd = Xb[trp].mean(0), Xb[trp].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[trp].T@Zt[trp] + 30*np.eye(Zt.shape[1]); A[-1,-1]-=30
w = np.linalg.solve(A, Zt[trp].T@y[trp])
resid = y - Zt@w
rows=[]
for j,c in enumerate(scols):
    x = Xs[:,j]
    if x.std()==0: continue
    rows.append((c, float(np.corrcoef(x, resid)[0,1])))
pc = pd.DataFrame(rows, columns=['f','pr']); pc['a']=pc.pr.abs()
print(pc.sort_values('a',ascending=False).to_string())