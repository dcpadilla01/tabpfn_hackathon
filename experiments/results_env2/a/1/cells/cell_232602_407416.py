import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
def run(Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)
hold=431
tr=day<hold; te=day==hold
w=0.5**((hold-day[tr])/140.0)
p=run(X[tr],y[tr],w,X[te])
r = y[te]-p
print("hold431 MAE", np.abs(r).mean().round(2), "bias(y-pred) mean", r.mean().round(2), "median", np.median(r).round(2))
# error by segment
dte = Ft[te].copy(); dte["p"]=p; dte["y"]=y[te]; dte["absr"]=np.abs(r)
dte["seg"] = pd.cut(dte.y, [-1,0.01,50,100,200,400,10000], labels=["0","0-50","50-100","100-200","200-400","400+"])
print(dte.groupby("seg",observed=True).agg(n=("absr","size"), mae=("absr","mean"), bias=("r","mean"), meanp=("p","mean"), meany=("y","mean")).round(1))
print("share of total MAE by seg:", (dte.groupby("seg",observed=True).absr.sum()/dte.absr.sum()).round(3))
# zero-target households: what does model predict?
z = dte[dte.y==0]
print("y=0: n", len(z), "mean pred", z.p.mean().round(1), "median pred", z.p.median().round(1), "their spend_28 mean", z.spend_28.mean().round(1))
# among y=0, mae by pred decile
print("MAE if we predicted 0 for all y=0 rows:", z.p.mean().round(2), "vs current", z.absr.mean().round(2))
