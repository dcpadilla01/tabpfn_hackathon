import agent_api as A
import pandas as pd, numpy as np, time

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[["med_v3","hgbq_v3","hgbq_all","oof_tw"]].values @ W0
y = oof.y.values; b = oof.blend.values; err = np.abs(b-y)

# error decomposition by y bucket
cuts = [0,1,10,50,100,200,500,1e9]
lab = ["y=0","0-10","10-50","50-100","100-200","200-500","500+"]
yb = pd.cut(y, bins=cuts, labels=lab, right=False)
df = pd.DataFrame({"yb":yb,"err":err,"b":b,"y":y})
g = df.groupby("yb", observed=True).apply(lambda t: pd.Series({
    "n":len(t), "share_of_MAE": t.err.sum()/err.sum(), "MAE": t.err.mean(),
    "pred_med": t.b.median(), "y_med": t.y.median()}), include_groups=False)
print(g.round(2))

# seed-averaging test on fold day=431: train XGB quantile on feats_v3, 3 seeds
fv = A.load_saved("feats_v3.parquet")
feats = [c for c in fv.columns if c not in ("household_key","snapshot_day")]
tr = fv.snapshot_day.values != 431
va = fv.snapshot_day.values == 431
ytr = tt.set_index(["household_key","snapshot_day"]).reindex(
    pd.MultiIndex.from_frame(fv[["household_key","snapshot_day"]])).values
ytr = ytr.ravel()
import xgboost as xgb
dtr = xgb.DMatrix(fv.loc[tr,feats], label=ytr[tr])
dva = xgb.DMatrix(fv.loc[va,feats])
ps=[]
t0=time.time()
for seed in [7, 42, 2024]:
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":6,
                   "learning_rate":0.03,"n_estimators":1200 if False else None,
                   "eval_metric":"mae","seed":seed,"tree_method":"hist"},
                  dtr, num_boost_round=1200, verbose_eval=False)
    ps.append(m.predict(dva))
    print("seed",seed,"fold431 MAE:", round(np.mean(np.abs(ps[-1]-ytr[va])),3), f"({time.time()-t0:.0f}s)")
print("avg of 3 seeds MAE:", round(np.mean(np.abs(np.mean(ps,axis=0)-ytr[va])),3))
print("med_v3 OOF fold431 MAE:", round(np.mean(np.abs(oof.loc[oof.snapshot_day==431,"med_v3"]-y[oof.snapshot_day.values==431])),3))
