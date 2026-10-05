import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

fv = A.load_saved("feats_v3.parquet")
fa = A.load_saved("feats_all_e016.parquet")   # v3 + seasonal
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
fv = fv.merge(tt, on=["household_key","snapshot_day"], how="left")
fa = fa.merge(tt, on=["household_key","snapshot_day"], how="left")
feats_v = [c for c in fv.columns if c not in ("household_key","snapshot_day","y")]
feats_a = [c for c in fa.columns if c not in ("household_key","snapshot_day","y")]
tr_all = fv.y.notna()
print("train rows:", tr_all.sum(), "val rows:", (~tr_all).sum())

def xgb_q(X, yv, Xp, cfg, seed=7):
    dtr = xgb.DMatrix(X, label=yv); dpr = xgb.DMatrix(Xp)
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"tree_method":"hist",
                   "seed":seed,"learning_rate":cfg[0],"max_depth":cfg[1]}, dtr,
                  num_boost_round=cfg[2], verbose_eval=False)
    return m.predict(dpr)

def hgb_q(X, yv, Xp, cfg, seed=7):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, random_state=seed,
        max_iter=cfg[0], learning_rate=cfg[1], max_leaf_nodes=31, min_samples_leaf=cfg[2],
        l2_regularization=0.1, early_stopping=False)
    m.fit(X, yv)
    return m.predict(Xp)

XCFG = {"a":(0.03,6,1200), "b":(0.02,7,2000), "c":(0.05,6,600)}
HCFG = {"a":(500,0.06,20), "b":(300,0.05,40)}

# ---- fold-431 sanity: pick best config per learner type
fold = 431
trf = tr_all & (fv.snapshot_day.values != fold)
vaf = tr_all & (fv.snapshot_day.values == fold)
oofc = A.load_saved("oof_e016_cv.parquet")
oofc = oofc.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oc431 = oofc[oofc.snapshot_day==fold].set_index(["household_key","snapshot_day"])
idx_va = pd.MultiIndex.from_frame(fv.loc[vaf,["household_key","snapshot_day"]])
y431 = fv.loc[vaf,"y"].values
ref = {"med_v3":np.mean(np.abs(oc431.loc[idx_va,"med_v3"]-y431)),
       "hgbq_v3":np.mean(np.abs(oc431.loc[idx_va,"hgbq_v3"]-y431)),
       "hgbq_all":np.mean(np.abs(oc431.loc[idx_va,"hgbq_all"]-y431))}
print("OOF fold431 refs:", {k:round(v,3) for k,v in ref.items()})

best_x = min(XCFG, key=lambda k: np.mean(np.abs(xgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], XCFG[k])-y431)))
print("best xgb cfg:", best_x, "MAE:", round(np.mean(np.abs(xgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], XCFG[best_x])-y431)),3))

best_hv = min(HCFG, key=lambda k: np.mean(np.abs(hgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], HCFG[k])-y431)))
print("best hgb(v3) cfg:", best_hv, "MAE:", round(np.mean(np.abs(hgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], HCFG[best_hv])-y431)),3))
best_ha = min(HCFG, key=lambda k: np.mean(np.abs(hgb_q(fa.loc[trf,feats_a], fa.loc[trf,"y"].values, fa.loc[vaf,feats_a], HCFG[k])-y431)))
print("best hgb(all) cfg:", best_ha, "MAE:", round(np.mean(np.abs(hgb_q(fa.loc[trf,feats_a], fa.loc[trf,"y"].values, fa.loc[vaf,feats_a], HCFG[best_ha])-y431)),3))

# ---- final: train on ALL 13 snapshots, predict val
t0=time.time()
pA = xgb_q(fv.loc[tr_all,feats_v], fv.loc[tr_all,"y"].values, fv.loc[~tr_all,feats_v], XCFG[best_x])
print("A done", f"{time.time()-t0:.0f}s")
pB = hgb_q(fv.loc[tr_all,feats_v], fv.loc[tr_all,"y"].values, fv.loc[~tr_all,feats_v], HCFG[best_hv])
print("B done", f"{time.time()-t0:.0f}s")
pC = hgb_q(fa.loc[tr_all,feats_a], fa.loc[tr_all,"y"].values, fa.loc[~tr_all,feats_a], HCFG[best_ha])
print("C done", f"{time.time()-t0:.0f}s")

pred = 0.4*pA + 0.3*pB + 0.3*pC
out = fv.loc[~tr_all,["household_key","snapshot_day"]].copy()
out["prediction"] = pred
print("val pred: n=",len(out)," mean=",pred.mean().round(2)," med=",np.median(pred).round(2)," max=",pred.max().round(1))

pe = A.load_saved("pred_e016.parquet").set_index(["household_key","snapshot_day"])["prediction"]
out_i = out.set_index(["household_key","snapshot_day"])
common = out_i.index.intersection(pe.index)
print("keys match E016:", len(common)==len(out))
d = out_i.loc[common,"prediction"] - pe.loc[common]
print("vs pred_e016: corr=", np.corrcoef(out_i.loc[common,"prediction"], pe.loc[common])[0,1].round(4),
      " mean diff=", d.mean().round(2), " MAE diff=", np.abs(d).mean().round(2))

path = A.save_table(out.reset_index(drop=True), "pred_e020")
print("saved:", path)
