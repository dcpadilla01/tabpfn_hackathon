import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]

base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
e001_cols = [c for c in e001.columns if c not in ("household_key","snapshot_day")]
cand_cols = [c for c in cand.columns if c not in ("household_key","snapshot_day")]

full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
# dedupe columns that appear twice (spend_28, nprod_28 etc.)
dupes = [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]
drop = set()
for c in dupes:
    stem = c[:-3]
    if stem in full.columns:
        a, b = full[stem].values.astype(float), full[c].values.astype(float)
        m = np.isnan(a)&np.isnan(b)
        if np.allclose(np.nan_to_num(a), np.nan_to_num(b)) or np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
        else:
            print("DIFFERS:", c)
full = full.drop(columns=list(drop))
print("full:", full.shape)

d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"

# rank features (cross-sectional within snapshot) for top features
rank_feats = {}
for c in ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]:
    if c in d.columns:
        rank_feats["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
d2 = pd.concat([d, pd.DataFrame(rank_feats)], axis=1)
rank_cols = list(rank_feats.columns)

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0):
    tr = d2[d2.snapshot_day.isin(tr_snaps)]; va = d2[d2.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]

e001_raw = [c for c in ["spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","weekly_rate_84","log_spend_all","log_spend_28"] if c in full.columns]
demo_cal = [c for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics","snapshot_day_index","week_of_year"] if c in full.columns]
best_new = ["a_std6","a_zero6","slope6","nb_84","top1_share_84","max_basket_84","ly_84","ly_56","avg_basket_84","items_pb_84","n_stores_84"]

print("\nlocal ridge MAE | holdout431 / holdout403  (BASE=E009 42 feats)")
b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"BASE          : {b1:.3f} / {b2:.3f}")
combos = {
 "+E001raw": base_cols+e001_raw,
 "+E001raw+demo/cal": base_cols+e001_raw+demo_cal,
 "+E001raw+bestnew": base_cols+e001_raw+best_new,
 "+E001raw+bestnew+demo/cal": base_cols+e001_raw+best_new+demo_cal,
 "+bestnew only": base_cols+best_new,
 "+demo/cal only": base_cols+demo_cal,
 "+ranks": base_cols+rank_cols,
 "+E001raw+ranks": base_cols+e001_raw+rank_cols,
 "+ALL": base_cols+e001_raw+best_new+demo_cal+rank_cols,
}
for name, feats in combos.items():
    m1 = ridge_eval(feats, trA, 431); m2 = ridge_eval(feats, trB, 403)
    print(f"{name:28s} n={len(feats):3d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")

# also check per-feature marginal value within BASE+E001raw via backward screen on 431
print("\nquick marginal check of best_new on top of BASE+E001raw (holdout431):")
f2 = base_cols+e001_raw
m0 = ridge_eval(f2, trA, 431)
for c in best_new:
    m = ridge_eval(f2+[c], trA, 431)
    print(f"  +{c:18s}: {m:.3f} ({m-m0:+.3f})")
