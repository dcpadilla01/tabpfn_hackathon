import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

def yearago_fn(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx.day
    def win(lo, hi):
        t = tx[(d > lo) & (d <= hi)]
        g = t.groupby("household_key")
        return g.sales_value.sum(), g.basket_id.nunique()
    s1,b1 = win(snapshot_day-363, snapshot_day-336)
    s2,b2 = win(snapshot_day-391, snapshot_day-336)
    s3,b3 = win(snapshot_day-727, snapshot_day-700)
    out = pd.DataFrame({"spend_y1_4w": s1, "baskets_y1_4w": b1, "spend_y1_8w": s2, "spend_y2_4w": s3}).reindex(hh).fillna(0.0)
    out["active_y1"] = (out.spend_y1_4w > 0).astype(float)
    return out

X = A.build_features(yearago_fn)          # 36426 x 7, indexed by household_key per snapshot
X = X.reset_index()                        # household_key, snapshot_day, feats
feats3 = A.load_saved("feats_v3.parquet")
f5 = feats3.merge(X, on=["household_key","snapshot_day"], how="inner")
print("f5", f5.shape)
A.save_table(f5, "feats_v5.parquet")

tt = A.train_targets()
df = f5.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in f5.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT: df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, cols, seed=7):
    ok = np.isfinite(tr.future_spend_4w.values)
    tr = tr.loc[ok]
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(tr[cols], tr.future_spend_4w)
    return m.predict(va[cols])

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]; vasnaps=[403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
yv = va.future_spend_4w.values
t0=time.time()
p0 = fit_pred(tr, va, FE)
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
p1 = fit_pred(tr, va, FE+NEW)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+yearago",p1)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())
