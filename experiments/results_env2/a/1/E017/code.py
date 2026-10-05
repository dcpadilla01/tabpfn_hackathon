names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds"]
preds = {}
for n in names:
    df = load_saved(f"{n}.parquet")
    preds[n] = df
    print(n, df.shape, list(df.columns), round(df.prediction.mean(),2), round(df.prediction.std(),2))

import itertools
base = preds["e005_preds"][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e005"})
print("base rows", base.shape)
for n in names[1:]:
    m = base.merge(preds[n][["household_key","snapshot_day","prediction"]], on=["household_key","snapshot_day"])
    c = np.corrcoef(m.prediction_x, m.prediction_y)[0,1]
    mad = np.abs(m.prediction_x-m.prediction_y).mean()
    print(n, "corr vs e005:", round(c,4), "mean|diff|:", round(mad,2))

oof = load_saved("oof_e5.parquet")
print("oof_e5:", oof.shape, list(oof.columns))
print(oof.head())


# ---- cell ----
names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds","e016_held","e016_allpreds"]
P = {}
for n in names:
    df = load_saved(f"{n}.parquet")
    P[n] = df
    print(n, df.shape, list(df.columns))

base = P["e005_preds"][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e005"})
for n in names[1:10]:
    m = base.merge(P[n].rename(columns={"prediction":n}), on=["household_key","snapshot_day"])
    c = np.corrcoef(m.e005, m[n])[0,1]
    mad = np.abs(m.e005-m[n]).mean()
    print(n, "corr:", round(c,4), "mean|diff|:", round(mad,2))

oof = load_saved("oof_e5.parquet")
print("oof_e5:", oof.shape, list(oof.columns))
print(oof.head())
print(oof.snapshot_day.unique() if "snapshot_day" in oof.columns else "")


# ---- cell ----
ap = load_saved("e016_allpreds.parquet")
print(ap.snapshot_day.value_counts().sort_index())
held = load_saved("e016_held.parquet")
print(held.snapshot_day.value_counts())
print(held.head())

# merge oof_e5 with allpreds on train rows
oof = load_saved("oof_e5.parquet")
m = oof.merge(ap, on=["household_key","snapshot_day"], how="inner")
print("merged train rows:", m.shape)
y = m.future_spend_4w.values
def mae(p): return np.abs(p-y).mean()
print("e5 OOF MAE:", round(mae(m.pred),3))
for c in ["pq","pl","pc","pa"]:
    print(c, "OOF MAE:", round(mae(m[c].values),3), "mean:", round(m[c].mean(),1), "std:", round(m[c].std(),1))
print("target mean:", round(y.mean(),1), "std:", round(y.std(),1))


# ---- cell ----
held = load_saved("e016_held.parquet")
names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds"]
print("Held (403,431) MAE by model:")
for n in names:
    m = held.merge(P[n], on=["household_key","snapshot_day"])
    print(n, round(np.abs(m.prediction-m.future_spend_4w).mean(),3))

ap = load_saved("e016_allpreds.parquet")
mh = held.merge(ap, on=["household_key","snapshot_day"])
for c in ["pq","pl","pc","pa"]:
    print("held", c, round(np.abs(mh[c]-mh.future_spend_4w).mean(),3))
# blends on held
def bl(*args):
    cs=[c for c in args]
    p = sum(mh[c] for c in cs)/len(cs)
    return np.abs(p-mh.future_spend_4w).mean()
print("blend pq+pl held:", round(bl("pq","pl"),3))
print("blend pq+pl+pa held:", round(bl("pq","pl","pa"),3))


# ---- cell ----
held = load_saved("e016_held.parquet")
names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds"]
print("Held (403,431) MAE by model:")
for n in names:
    m = held.merge(load_saved(f"{n}.parquet"), on=["household_key","snapshot_day"])
    print(n, round(np.abs(m.prediction-m.future_spend_4w).mean(),3))

ap = load_saved("e016_allpreds.parquet")
mh = held.merge(ap, on=["household_key","snapshot_day"])
for c in ["pq","pl","pc","pa"]:
    print("held", c, round(np.abs(mh[c]-mh.future_spend_4w).mean(),3))
def bl(*cs):
    p = sum(mh[c] for c in cs)/len(cs)
    return np.abs(p-mh.future_spend_4w).mean()
print("blend pq+pl held:", round(bl("pq","pl"),3))
print("blend pq+pl+pa held:", round(bl("pq","pl","pa"),3))


# ---- cell ----
held = load_saved("e016_held.parquet")
e5 = load_saved("e005_preds.parquet")
print(held.dtypes, e5.dtypes)
print(held.household_key.head().tolist(), e5.household_key.head().tolist())
print(held.snapshot_day.unique(), e5.snapshot_day.unique())
m = held.merge(e5, on=["household_key","snapshot_day"])
print(m.shape, m.prediction.isna().sum())
# check key sets
s5 = set(map(tuple, e5[["household_key","snapshot_day"]].values))
sh = set(map(tuple, held[["household_key","snapshot_day"]].values))
print("overlap:", len(s5 & sh), "held:", len(sh), "e5:", len(s5))
print(list(sh)[:5], list(s5)[:5])


# ---- cell ----
oof = load_saved("oof_e5.parquet")
ap = load_saved("e016_allpreds.parquet")
held = load_saved("e016_held.parquet")

# target mean by snapshot day (train)
print(oof.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]).round(1))

m = oof.merge(ap[["household_key","snapshot_day","pq","pl","pa"]], on=["household_key","snapshot_day"])
print("\nOOF bias (pred-target) by day:")
print(m.groupby("snapshot_day").apply(lambda g: pd.Series({
    "tgt": g.future_spend_4w.mean(), "pq": g.pq.mean(), "pl": g.pl.mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean(), "pl_bias": (g.pl-g.future_spend_4w).mean()}), include_groups=False).round(1))

mh = held.merge(ap[["household_key","snapshot_day","pq","pl","pa"]], on=["household_key","snapshot_day"])
print("\nHeld bias by day:")
print(mh.groupby("snapshot_day").apply(lambda g: pd.Series({
    "tgt": g.future_spend_4w.mean(), "pq": g.pq.mean(), "pl": g.pl.mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean()}), include_groups=False).round(1))

# MAE by target bucket on held for pq
mh["bucket"] = pd.cut(mh.future_spend_4w, [-1,1,50,100,200,400,10000])
print("\nHeld MAE by target bucket:")
print(mh.groupby("bucket", observed=True).apply(lambda g: pd.Series({
    "n": len(g), "pq": np.abs(g.pq-g.future_spend_4w).mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean()}), include_groups=False).round(1))


# ---- cell ----
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ap = load_saved("e016_allpreds.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]

tr = ap.merge(oof, on=["household_key","snapshot_day"])
he = ap.merge(held, on=["household_key","snapshot_day"])
print("tr", tr.shape, "days", sorted(tr.snapshot_day.unique()))
print("he", he.shape, "days", sorted(he.snapshot_day.unique()))
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()
print("pq train MAE (in-sample?):", round(mae(tr.pq, tr.future_spend_4w),3))
print("pq held  MAE:", round(mae(he.pq, he.future_spend_4w),3))

ytr, ptr, dtr = tr.future_spend_4w.values, tr.pq.values, tr.snapshot_day.values
yhe, phe, dhe = he.future_spend_4w.values, he.pq.values, he.snapshot_day.values

# C1 constant shift (MAE-optimal = median residual)
c1 = np.median(ytr - ptr)
# C2 multiplicative: grid
grid = np.arange(0.95, 1.35, 0.005)
maes = [mae(a*ptr, ytr) for a in grid]
a2 = grid[int(np.argmin(maes))]
# C3 isotonic target|pred
iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
# C4 isotonic + linear day-trend of residual
res = ytr - iso.predict(ptr)
daymed = tr.assign(r=res).groupby("snapshot_day").r.median()
sl, ic = np.polyfit(daymed.index.values, daymed.values, 1)
print("day-trend of residual: slope %.4f intercept %.2f" % (sl, ic))
# C5 binned median residual vs pred (non-monotone correction)
qb = np.quantile(ptr, np.linspace(0,1,21)); qb[0]-=1; qb[-1]+=1
bins = pd.cut(ptr, qb)
bmed = tr.assign(r=res).groupby(bins, observed=True).r.median()
bctr = np.array([(b.left+b.right)/2 for b in bmed.index])
# C6 isotonic weighted toward later days (half-life 140d from day 375)
w = 0.5**((375-dtr)/140)
iso6 = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr, sample_weight=w)

def apply_c5(p):
    corr = np.interp(p, bctr, bmed.values)
    return p + corr

cands = {
 "C0 identity": phe,
 "C1 const+%.1f"%c1: phe+c1,
 "C2 mult %.3f"%a2: a2*phe,
 "C3 isotonic": iso.predict(phe),
 "C4 iso+daytrend": iso.predict(phe) + (sl*dhe+ic),
 "C5 binnedresid": apply_c5(phe),
 "C6 iso_wdecay": iso6.predict(phe),
}
print("\nHELD MAE (fit on train 95-375):")
for k,v in cands.items(): print(" ", k, round(mae(v,yhe),3))

# temporal honesty check: fit on 95-291, eval on 319-375
mtr = dtr <= 291; mev = dtr >= 319
yt, pt = ytr[mtr], ptr[mtr]; ye, pe = ytr[mev], ptr[mev]
c1t = np.median(yt-pt)
isoT = IsotonicRegression(out_of_bounds="clip").fit(pt, yt)
resT = yt - isoT.predict(pt)
dmedT = tr[mtr].assign(r=resT).groupby("snapshot_day").r.median()
slT, icT = np.polyfit(dmedT.index.values, dmedT.values, 1)
print("\nTemporal check (fit 95-291 -> eval 319-375):")
print("  identity:", round(mae(pe,ye),3))
print("  const:", round(mae(pe+c1t,ye),3))
print("  isotonic:", round(mae(isoT.predict(pe),ye),3))
print("  iso+daytrend:", round(mae(isoT.predict(pe)+slT*319+icT, ye),3), "(day set to 319)")


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
allF = load_saved("allF.parquet")
print("allF:", allF.shape)
print(allF.snapshot_day.value_counts().sort_index())
print("cols sample:", list(allF.columns)[:10], "...")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day")]
print("n_feats:", len(feats))
tr = allF.merge(oof, on=["household_key","snapshot_day"])
he = allF.merge(held, on=["household_key","snapshot_day"])
print("tr:", tr.shape, "he:", he.shape)
print("tr days:", sorted(tr.snapshot_day.unique()), "he days:", sorted(he.snapshot_day.unique()))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

allF = load_saved("allF.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day")]
tr = allF.merge(oof, on=["household_key","snapshot_day"])
he = allF.merge(held, on=["household_key","snapshot_day"])
Xtr, Xhe = tr[feats].values.astype(np.float32), he[feats].values.astype(np.float32)
ytr, yhe = tr.future_spend_4w.values, he.future_spend_4w.values
dtr, dhe = tr.snapshot_day.values, he.snapshot_day.values
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()

def qw(y): return 0.5**((375.0-y)/140.0)

t0=time.time()
params = dict(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
              learning_rate=0.03, max_depth=6, min_child_weight=25, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5, base_score=0.5,
              n_jobs=4, eval_metric="mae")
preds_he, preds_tr = [], []
for s in [7,17,27]:
    d = xgb.DMatrix(Xtr, label=ytr, weight=qw(ytr))
    b = xgb.train(params, d, num_boost_round=1200, verbose_eval=False)
    preds_he.append(b.predict(xgb.DMatrix(Xhe)))
    preds_tr.append(b.predict(xgb.DMatrix(Xtr)))
    print("seed", s, "held MAE:", round(mae(preds_he[-1], yhe),3), round(time.time()-t0,1),"s")
Phe = np.mean(preds_he, axis=0); Ptr = np.mean(preds_tr, axis=0)
print("ENSEMBLE held MAE:", round(mae(Phe, yhe),3), "| E005 held MAE 62.82")
print("ENSEMBLE train-fit MAE:", round(mae(Ptr, ytr),3))
print("mean pred held:", round(Phe.mean(),1), "target mean:", round(yhe.mean(),1))


# ---- cell ----
import numpy as np, pandas as pd
oof = load_saved("oof_e5.parquet")
held = load_saved("e016_held.parquet")
print("oof cols:", list(oof.columns))
print("held cols:", list(held.columns))
print("allF cols tail:", [c for c in load_saved("allF.parquet").columns][-8:])


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

allF = load_saved("allF.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
# check allF target on train rows matches oof_e5
chk = allF.merge(oof, on=["household_key","snapshot_day"], suffixes=("_F","_oof"))
print("target match:", np.allclose(chk.future_spend_4w_F, chk.future_spend_4w_oof))
valF = allF[allF.snapshot_day>=459]
print("allF val target NaNs:", valF.future_spend_4w.isna().mean())

tr = allF.merge(oof, on=["household_key","snapshot_day"])
he = allF.merge(held, on=["household_key","snapshot_day"])
Xtr, Xhe = tr[feats].values.astype(np.float32), he[feats].values.astype(np.float32)
ytr, yhe = tr.future_spend_4w_oof.values, he.future_spend_4w.values
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()
def qw(y): return 0.5**((375.0-y)/140.0)

params = dict(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
              learning_rate=0.03, max_depth=6, min_child_weight=25, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5, base_score=0.5,
              n_jobs=4, eval_metric="mae")
t0=time.time()
preds_he, preds_tr = [], []
for s in [7,17,27]:
    d = xgb.DMatrix(Xtr, label=ytr, weight=qw(ytr))
    b = xgb.train(params, d, num_boost_round=1200, verbose_eval=False)
    preds_he.append(b.predict(xgb.DMatrix(Xhe)))
    preds_tr.append(b.predict(xgb.DMatrix(Xtr)))
    print("seed", s, "held MAE:", round(mae(preds_he[-1], yhe),3), round(time.time()-t0,1),"s")
Phe = np.mean(preds_he, axis=0); Ptr = np.mean(preds_tr, axis=0)
print("ENSEMBLE held MAE:", round(mae(Phe, yhe),3), "| E005 held MAE 62.82")
print("train-fit MAE:", round(mae(Ptr, ytr),3), "| mean pred held:", round(Phe.mean(),1), "tgt:", round(yhe.mean(),1))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

allF = load_saved("allF.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
tgt = allF[["household_key","snapshot_day","future_spend_4w"]].rename(columns={"future_spend_4w":"tgt_F"})
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]

tr = allF.drop(columns=["future_spend_4w"]).merge(oof, on=["household_key","snapshot_day"])
he = allF.drop(columns=["future_spend_4w"]).merge(held, on=["household_key","snapshot_day"])
val = allF[allF.snapshot_day>=459]
Xtr, Xhe, Xval = (tr[feats].values.astype(np.float32), he[feats].values.astype(np.float32),
                  val[feats].values.astype(np.float32))
ytr, yhe = tr.future_spend_4w.values, he.future_spend_4w.values
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()
def qw(y): return 0.5**((375.0-y)/140.0)
params = dict(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
              learning_rate=0.03, max_depth=6, min_child_weight=25, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5, base_score=0.5,
              n_jobs=4, eval_metric="mae")
t0=time.time()
preds_he, preds_val = [], []
for s in [7,17,27]:
    d = xgb.DMatrix(Xtr, label=ytr, weight=qw(ytr))
    b = xgb.train(params, d, num_boost_round=1200, verbose_eval=False)
    preds_he.append(b.predict(xgb.DMatrix(Xhe)))
    preds_val.append(b.predict(xgb.DMatrix(Xval)))
    print("seed", s, "held MAE:", round(mae(preds_he[-1], yhe),3), round(time.time()-t0,1),"s")
Phe = np.mean(preds_he, axis=0); Pval = np.mean(preds_val, axis=0)
print("ENSEMBLE held MAE:", round(mae(Phe, yhe),3), "| E005 held 62.82")
print("val preds:", Pval.shape, "mean", round(Pval.mean(),1), "finite:", np.isfinite(Pval).all())
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = Pval.astype(np.float64)
out = out.reset_index(drop=True)
print(out.shape, out.snapshot_day.value_counts().to_dict())
path = save_table(out, "e017_preds")
print("saved:", path)
