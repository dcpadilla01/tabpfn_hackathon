import numpy as np, pandas as pd

tt = train_targets()
def loso(tab, label, alpha=50.0):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    # drop degenerate columns
    keep = X.columns[(X.nunique(dropna=True)>1).values]
    X = X[keep]
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = np.clip((X.values-mu)/sd, -5, 5)
    errs=[]
    for s in sorted(set(days))[:13]:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + alpha*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = np.clip(Xs[te]@w, 0, 1500)
        errs.append(np.abs(pred-y[te]).mean())
    print(f"{label}: LOSO-MAE {np.mean(errs):.3f} (n_feat {X.shape[1]})")
    return np.mean(errs)

a = load_saved("e019_everything.parquet")
m = load_saved("e018_union_full.parquet")
loso(a, "e019_everything (E017 best)")
loso(m, "union (+31 timing)")