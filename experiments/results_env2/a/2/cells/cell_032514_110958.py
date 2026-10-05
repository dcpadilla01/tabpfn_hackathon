import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb

fv = A.load_saved("feats_v3.parquet")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
fv = fv.merge(tt, on=["household_key","snapshot_day"], how="left")
print("rows:", len(fv), "y NaN (val rows):", fv.y.isna().sum())
feats = [c for c in fv.columns if c not in ("household_key","snapshot_day","y")]
fv_tr = fv[fv.y.notna()]

tr = fv_tr.snapshot_day.values != 431
va = fv_tr.snapshot_day.values == 431
dtr = xgb.DMatrix(fv_tr.loc[tr,feats], label=fv_tr.loc[tr,"y"].values)
dva = xgb.DMatrix(fv_tr.loc[va,feats])
ps=[]; t0=time.time()
for seed in [7, 42, 2024]:
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":6,
                   "learning_rate":0.03,"seed":seed,"tree_method":"hist"},
                  dtr, num_boost_round=1200, verbose_eval=False)
    ps.append(m.predict(dva))
    print("seed",seed,"fold431 MAE:", round(np.mean(np.abs(ps[-1]-fv_tr.loc[va,"y"].values)),3), f"({time.time()-t0:.0f}s)")
print("avg of 3 seeds MAE:", round(np.mean(np.abs(np.mean(ps,axis=0)-fv_tr.loc[va,"y"].values)),3))
