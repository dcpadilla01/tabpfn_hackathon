import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")

tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]], on=["household_key","snapshot_day"], how="left")

feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
# exact duplicate columns
Xd = m2[feats].copy()
for c in feats:
    if Xd[c].dtype == object: Xd[c] = pd.factorize(Xd[c])[0]
Xd = Xd.apply(pd.to_numeric, errors="coerce").fillna(0.0)
dupmask = Xd.T.duplicated().values
dups = [f for f,d in zip(feats,dupmask) if d]
print("exact dup cols (%d):"%len(dups), dups)

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

# greedy forward selection on proxy (fit<=403, eval@431), starting from empty, candidate pool = strongest 40 by |spear|
y_all = m2.future_spend_4w.values
pool = []
for c in feats + ["oh_ewm1.5","oh_ewm4","oh_n"]:
    s = m2[c]
    if s.dtype == object: s = pd.factorize(s)[0]
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    pool.append((abs(s.corr(m2.future_spend_4w, method="spearman")), c))
pool.sort(reverse=True)
cands = [c for _,c in pool[:45]]
print("top45:", cands[:15], "...")
sel = []
cur = 1e9
for it in range(14):
    best = None
    for c in cands:
        if c in sel: continue
        e = ridge_eval(sel+[c], 403, 431)
        if best is None or e < best[1]: best = (c, e)
    if best[1] < cur - 1e-4:
        sel.append(best[0]); cur = best[1]
        print(f"step{it+1}: +{best[0]:26s} MAE={best[1]:.3f}  (n={len(sel)})")
    else:
        print("no improvement; stop"); break
print("\nfinal sel:", sel, "MAE=", round(cur,3))
print("full base+oh:", round(ridge_eval(feats+["oh_ewm1.5","oh_ewm4","oh_n"],403,431),3))
print("base dedup:", round(ridge_eval([f for f in feats if f not in dups],403,431),3))
