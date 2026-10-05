import agent_api as A, pandas as pd, numpy as np, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
def train_one(obj,Xtr,ytr,wtr,rounds=1200,seeds=(11,22,33),lr=0.03,md=6):
    ps = dict(objective=obj,quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile" if obj.startswith("reg:quantile") else "mae")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(xgb.DMatrix(Xtr if Xte is None else Xte)))
    return np.mean(pr,axis=0)
def train_one(obj,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33),lr=0.03,md=6):
    ps = dict(objective=obj,quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile" if "quantile" in obj else "mae")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)

hold=431; tr=day<hold; te=day==hold
w = 0.5**((hold-day[tr])/140.0)
pq = train_one("reg:quantileerror",X[tr],y[tr],w,X[te])
pm = train_one("reg:squarederror",X[tr],y[tr],w,X[te])
yv=y[te]
print("MAE quantile", np.abs(pq-yv).mean().round(2), " MAE sqerr", np.abs(pm-yv).mean().round(2))
for wq in [0.8,0.7,0.6,0.5]:
    p = wq*pq+(1-wq)*pm
    print(f"blend wq={wq}: MAE", np.abs(p-yv).mean().round(2))
# log-target squared error with smearing
yl = np.log1p(y[tr])
pl = train_one("reg:squarederror",X[tr],yl,w,X[te])
resid = yl - None
# smearing factor from train
dtr=xgb.DMatrix(X[tr],label=yl,weight=w)
ps=dict(objective="reg:squarederror",learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist")
b=xgb.train(ps,dtr,1200); intr=b.predict(dtr)
fac = np.mean(np.exp(yl-intr))
plog = np.expm1(pl)*fac
print("MAE log-model smeared:", np.abs(plog-yv).mean().round(2))
for wq in [0.7,0.5]:
    p=wq*pq+(1-wq)*plog
    print(f"blend q+log wq={wq}: MAE", np.abs(p-yv).mean().round(2))
