import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
print("F", F.shape)
tt = A.train_targets()
Ft = F.merge(tt, on=["household_key","snapshot_day"])
print("Ft", Ft.shape)
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
# extra cheap ratio features
Ft["r28_364"] = Ft.spend_28/(Ft.spend_364/13+1e-6)
Ft["r7_364"]  = Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"] = Ft.spend_84/(Ft.spend_364/4+1e-6)
Ft["act_ratio_28_84"] = Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
feats = [c for c in feats if c in Ft.columns]
Xall = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values

def run_xgb(Xtr,ytr,wtr,Xte,params,rounds,seeds=(11,22,33)):
    dtr = xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte = xgb.DMatrix(Xte)
    ps = dict(params); ps["objective"]="reg:quantileerror"; ps["quantile_alpha"]=0.5
    preds=[]
    for s in seeds:
        ps["seed"]=s
        b = xgb.train(ps,dtr,rounds)
        preds.append(b.predict(dte))
    return np.mean(preds,axis=0)

base = dict(learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
t0=time.time()
for hold in [347,403,431]:
    tr = day<hold; te = day==hold
    w = 0.5**((hold-day[tr])/140.0)
    p = run_xgb(Xall[tr],y[tr],w,Xall[te],base,1200)
    mae = np.abs(p-y[te]).mean()
    print(f"hold {hold}: n_tr={tr.sum()} MAE={mae:.3f}  ({time.time()-t0:.0f}s)")
