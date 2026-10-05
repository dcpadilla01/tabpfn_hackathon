import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
# global mean target by day: 130->146 rising. Try adding snapshot_day itself as feature
Ft["sd"] = Ft.snapshot_day
feats2 = feats+["sd"]

def run(Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33),lr=0.03,md=6,mcw=5):
    ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=mcw,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)

for featsX,name in [(feats,"noSD"),(feats2,"withSD")]:
    Xf = Ft[featsX].astype("float32")
    maes=[]
    for hold in [347,403,431]:
        tr=day<hold; te=day==hold
        w=0.5**((hold-day[tr])/140.0)
        p=run(Xf[tr],y[tr],w,Xf[te])
        maes.append(np.abs(p-y[te]).mean())
    print(name, [round(m,2) for m in maes], "avg", round(np.mean(maes),3))
