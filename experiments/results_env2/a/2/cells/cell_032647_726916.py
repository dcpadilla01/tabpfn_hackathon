import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
oof = oof.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
fv = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
Xf = fv.merge(fs, on=["household_key","snapshot_day"], how="left")
key_feats = ["spend_all","spend_28","spend_56","spend_84","baskets_28","spend_max_basket_all","lag364_spend","lag308_spend"]
oof = oof.merge(Xf[["household_key","snapshot_day"]+key_feats], on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[L].values @ W0
y = oof.y.values; D = oof.snapshot_day.values; blend = oof.blend.values

# cheap: piecewise multiplicative shrink/expand by blend bucket, median-objective per bucket, LOSO
def fp_pw(tr,va):
    b_tr = blend[tr]; y_tr = y[tr]
    edges = [0,25,75,150,300,600,np.inf]
    corr = []
    for i in range(len(edges)-1):
        m = (b_tr>=edges[i]) & (b_tr<edges[i+1])
        if m.sum()<50: corr.append(1.0); continue
        # median of y/b ratio in bucket
        r = y_tr[m]/np.maximum(b_tr[m],1e-6)
        corr.append(np.clip(np.median(r),0.5,2.0))
    b_va = blend[va]
    idx = np.digitize(b_va, edges[1:-1])
    return b_va * np.array(corr)[idx]
maes=[]
for d in days:
    tr = D!=d; va=~tr
    maes.append(np.mean(np.abs(fp_pw(tr,va)-y[va])))
print("piecewise ratio LOSO:", round(np.mean(maes),4))

# stage-2 XGB quantile on [learners + blend + key feats]
s2cols = L + ["blend"] + key_feats
def fp_s2(tr,va,seed=7,rounds=800,lr=0.05,depth=5):
    dtr = xgb.DMatrix(oof.loc[tr,s2cols], label=y[tr])
    dva = xgb.DMatrix(oof.loc[va,s2cols])
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":depth,
                   "learning_rate":lr,"seed":seed,"tree_method":"hist"}, dtr, num_boost_round=rounds, verbose_eval=False)
    return m.predict(dva)
maes=[]; t0=time.time()
for d in days:
    tr = D!=d; va=~tr
    maes.append(np.mean(np.abs(fp_s2(tr,va)-y[va])))
print("stage2 XGBq LOSO:", round(np.mean(maes),4), [round(x,2) for x in maes], f"({time.time()-t0:.0f}s)")
print("ref W0 LOSO:", round(np.mean([np.mean(np.abs(blend[D!=d][~(D==d)]-y[D!=d])) for d in days]),4))
