import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
F["r28_364"]=F.spend_28/(F.spend_364/13+1e-6); F["r7_364"]=F.spend_7/(F.spend_364/50+1e-6)
F["r84_364"]=F.spend_84/(F.spend_364/4+1e-6); F["act_ratio_28_84"]=F.act_28/(F.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
tr_rows = F.snapshot_day<=431; va_rows = F.snapshot_day>=459
Xtr = F.loc[tr_rows, feats].astype("float32").values
w = 0.5**((459 - F.loc[tr_rows,"snapshot_day"].values)/140.0)
tt = A.train_targets(); ytr = F.loc[tr_rows,["household_key","snapshot_day"]].merge(tt, on=["household_key","snapshot_day"]).future_spend_4w.values.astype("float32")
assert len(ytr)==tr_rows.sum()
Xva = F.loc[va_rows, feats].astype("float32").values
ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
dtr=xgb.DMatrix(Xtr,label=ytr,weight=w); dva=xgb.DMatrix(Xva)
t0=time.time(); pr=[]
for s in (11,22,33,44,55):
    ps["seed"]=s; b=xgb.train(ps,dtr,2400); pr.append(b.predict(dva))
    print("seed",s,"done",round(time.time()-t0))
pred = np.mean(pr,axis=0)
out = F.loc[va_rows,["household_key","snapshot_day"]].copy(); out["prediction"]=pred
print(out.shape, out.prediction.describe().round(1).to_dict())
p = A.save_table(out, "e007_preds")
print("path:", p)
