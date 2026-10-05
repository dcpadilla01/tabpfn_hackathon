import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
f3 = A.load_saved("feats_v3.parquet"); fs = A.load_saved("feats_seasonal.parquet")
F = f3.merge(fs, on=["household_key","snapshot_day"], how="left")
tt = A.train_targets()
data = F.merge(tt, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
feat_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
X = data[feat_cols].astype(float).values
y = data.future_spend_4w.values
day = data.snapshot_day.values

def mk(obj, n, lr, seed=7):
    return xgb.XGBRegressor(objective=obj, n_estimators=n, learning_rate=lr, max_depth=7,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, random_state=seed)

oof_days=[347,375,403,431]
t0=time.time(); out={}
cfgs = [("med", lambda: mk("reg:absoluteerror",1200,0.03), None),
        ("sq",  lambda: mk("reg:squarederror",2000,0.02), None),
        ("med_w112", lambda: mk("reg:absoluteerror",1200,0.03), 112.),
        ("med_w224", lambda: mk("reg:absoluteerror",1200,0.03), 224.)]
oof_store = {}
for name, mkf, H in cfgs:
    oof = np.full(len(data), np.nan)
    for d in oof_days:
        tr = day < d; te = day == d
        w = None
        if H is not None:
            ref = day[tr].max()
            w = 0.5 ** ((ref - day[tr]) / H)
        m = mkf(); m.fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict(X[te])
    mae = np.nanmean(np.abs(oof-y))
    out[name]=round(mae,3); oof_store[name]=oof
    print(name, "OOF MAE:", round(mae,3), f"({time.time()-t0:.0f}s)", flush=True)

# save oof for stacking later
odf = data[["household_key","snapshot_day"]].copy()
for k,v in oof_store.items(): odf["oof_"+k]=v
p = A.save_table(odf, "oof_harness.parquet")
print("saved:", p, "| MAEs:", out)
