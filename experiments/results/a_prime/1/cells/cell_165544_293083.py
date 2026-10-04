import numpy as np, pandas as pd

# Build the TRUE union: E017 best table (e019_everything) + E016's 33 trip-timing features (e018_timing_hazard)
a = load_saved("e019_everything.parquet")
b = load_saved("e018_timing_hazard.parquet")
timing_cols = [c for c in b.columns if c not in a.columns and c not in ("index",)]
print("timing cols to add:", len(timing_cols), timing_cols)
m = a.merge(b[["household_key","snapshot_day"]+timing_cols], on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
m = m.drop(columns=[c for c in m.columns if c=="index"], errors="ignore")
print("dupes:", m.duplicated(["household_key","snapshot_day"]).sum())
p = save_table(m, "e018_union_full")
print("saved:", p)

# --- internal sanity: ridge LOSO on TRAIN snapshots only, compare e019 vs union ---
tt = train_targets()
def loso(tab, label):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    tr_days = sorted([s for s in d.snapshot_day.unique()])[:13]
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = (X.values-mu)/sd
    errs=[]
    for s in tr_days:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + 3.0*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = Xs[te]@w
        errs.append(np.abs(pred-y[te]).mean())
    print(f"{label}: LOSO-MAE {np.mean(errs):.3f}")
    return np.mean(errs)

loso(a, "e019_everything (E017 best)")
loso(m, "union (+33 timing)")