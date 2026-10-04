import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]

full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
drop = set()
for c in [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]:
    stem = c[:-3]
    if stem in full.columns:
        a = full[stem].values.astype(float); b = full[c].values.astype(float)
        if np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
full = full.drop(columns=list(drop))
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
num_cols, cat_cols = [], []
for c in full.columns:
    if c in ("household_key","snapshot_day"): continue
    (num_cols if pd.api.types.is_numeric_dtype(d[c]) else cat_cols).append(c)
enc = pd.DataFrame(index=d.index)
for c in cat_cols:
    enc[c] = pd.factorize(d[c].astype(str))[0].astype(float)
enc = pd.concat([enc, d[num_cols].astype(float)], axis=1)

rank_srcs = ["spend_84","spend_28","e13","e6","usual13","usual6","b75","p13","p6","hazard","dsl","rvu","spend_182","nb_84","nprod_28","gap_n21","cv13"]
rk = pd.DataFrame(index=d.index)
for c in rank_srcs:
    if c in enc.columns:
        rk["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
rk_cols = list(rk.columns)
small = ["rk_spend_84","rk_spend_28","rk_e13","rk_usual13","rk_b75","rk_nb_84","rk_nprod_28"]
enc3 = pd.concat([enc, rk], axis=1)

def ridge_eval(feats, tr_snaps, val_snap, Xdf, alpha=100.0):
    tr = Xdf.loc[d.snapshot_day.isin(tr_snaps).values]
    va = Xdf.loc[(d.snapshot_day==val_snap).values]
    Xtr = tr[feats].values.copy(); Xva = va[feats].values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = d.loc[tr.index, ycol].values.astype(float); yva = d.loc[va.index, ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
b1 = ridge_eval(base_cols, trA, 431, enc); b2 = ridge_eval(base_cols, trB, 403, enc)
print(f"BASE: {b1:.3f} / {b2:.3f} avg={(b1+b2)/2:.3f}")
for name, rc in [("+17ranks", rk_cols), ("+7ranks", small)]:
    m1 = ridge_eval(base_cols+rc, trA, 431, enc3); m2 = ridge_eval(base_cols+rc, trB, 403, enc3)
    print(f"{name}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f}) avg={(m1+m2)/2:.3f}")

# greedy backward pruning of base features (ranks kept fixed = 7 small ranks)
cur = list(base_cols)
curA = ridge_eval(cur+small, trA, 431, enc3); curB = ridge_eval(cur+small, trB, 403, enc3)
print(f"\nstart cur+7ranks: {curA:.3f}/{curB:.3f} avg={(curA+curB)/2:.3f}")
for it in range(5):
    scores = []
    for c in cur:
        f = [x for x in cur if x != c]
        m1 = ridge_eval(f+small, trA, 431, enc3); m2 = ridge_eval(f+small, trB, 403, enc3)
        scores.append((c, (m1+m2)/2-(curA+curB)/2, m1-curA, m2-curB))
    sdf = pd.DataFrame(scores, columns=["c","davg","d1","d2"]).sort_values("davg")
    top = sdf.iloc[0]
    if top.davg < -0.03 and top.d1 < 0.05 and top.d2 < 0.05:
        dropped = top.c
        cur = [x for x in cur if x != dropped]
        curA = ridge_eval(cur+small, trA, 431, enc3); curB = ridge_eval(cur+small, trB, 403, enc3)
        print(f"drop {dropped:12s} -> {curA:.3f}/{curB:.3f} avg={(curA+curB)/2:.3f}")
    else:
        print("stop: no more good drops"); break
print("removed:", [c for c in base_cols if c not in cur])

final_feats = cur + small
fA = ridge_eval(final_feats, trA, 431, enc3); fB = ridge_eval(final_feats, trB, 403, enc3)
print(f"\nFINAL ({len(final_feats)} feats): {fA:.3f} / {fB:.3f} avg={(fA+fB)/2:.3f}  [BASE avg {(b1+b2)/2:.3f}]")

# build & save final table: e009 base (pruned) + within-snapshot ranks from e009 itself
out = e009[["household_key","snapshot_day"]+cur].copy()
for c in small:
    src = c[3:]
    out[c] = e009.groupby("snapshot_day")[src].rank(pct=True).values
A.save_table(out, "e013_ranks.parquet")
print("saved e013_ranks.parquet", out.shape)
