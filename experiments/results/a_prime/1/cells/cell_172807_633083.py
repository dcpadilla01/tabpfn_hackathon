import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"])
y = tr.future_spend_4w.values.astype(float)
# 1) log-space ridge blend of top ~40 features (no tree model; ridge on log1p(y))
from numpy.linalg import lstsq
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
numcols = [c for c in numcols if tr[c].nunique(dropna=True)>1]
# standardize, impute median
Z = tr[numcols].astype(float)
med = Z.median()
Z = Z.fillna(med).values
Z = (Z - Z.mean(0)) / (Z.std(0)+1e-9)
Z = np.clip(Z, -8, 8)
ly = np.log1p(y)
# ridge sweep with alpha
def ridge_fit(Zt, yt, alpha):
    n,p = Zt.shape
    A_ = Zt.T@Zt + alpha*np.eye(p)
    return np.linalg.solve(A_, Zt.T@yt)
# leave-one-snapshot-out CV
days = sorted(tr.snapshot_day.unique())
def loso(alpha, target="log"):
    preds = np.zeros(len(tr))
    for d in days:
        te = (tr.snapshot_day.values==d)
        Ztr, ytr = Z[~te], (ly[~te] if target=="log" else y[~te])
        w = ridge_fit(Ztr, ytr, alpha)
        preds[te] = Z[te]@w
        if target=="log": preds[te] = np.expm1(np.clip(preds[te],0,8))
    return np.mean(np.abs(y-preds))
for alpha in [30,100,300,1000]:
    print("alpha",alpha,"log-ridge LOSO MAE", round(loso(alpha,"log"),2), "| linear LOSO MAE", round(loso(alpha,"lin"),2))