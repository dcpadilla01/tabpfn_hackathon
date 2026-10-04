import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
y = tr.future_spend_4w.values.astype(float)
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
numcols = [c for c in numcols if tr[c].nunique(dropna=True)>1]
Z = tr[numcols].astype(float)
med = Z.median(); Z = Z.fillna(med).values
Z = np.clip((Z - Z.mean(0))/(Z.std(0)+1e-9), -8, 8)
ly = np.log1p(y)
def ridge(Zt, yt, alpha):
    p = Zt.shape[1]
    return np.linalg.solve(Zt.T@Zt + alpha*np.eye(p), Zt.T@yt)
def loso(alpha, mode):
    preds = np.zeros(len(tr))
    for d in days:
        te = (tr.snapshot_day.values == d)
        w = ridge(Z[~te], (ly if mode=="log" else y)[~te], alpha)
        p = Z[te]@w
        if mode=="log": p = np.expm1(np.clip(p,0,8))
        preds[te] = p
    return np.mean(np.abs(y-preds))
for alpha in [10,30,100,300,1000,3000]:
    print(alpha, "log", round(loso(alpha,"log"),2), "lin", round(loso(alpha,"lin"),2))