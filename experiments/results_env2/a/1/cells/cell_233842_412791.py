import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
t0=time.time()
def pred(ps,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    ps=dict(ps); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds); pr.append(b.predict(xgb.DMatrix(Xte)))
    return np.mean(pr,axis=0)
hold=431; tr=day<hold; te=day==hold; yv=y[te]
w=0.5**((hold-day[tr])/140.0)
base=dict(learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist")
pq = pred({**base,"objective":"reg:quantileerror","quantile_alpha":0.5,"eval_metric":"quantile"},X[tr],y[tr],w,X[te])
pb = pred({**base,"objective":"binary:logistic","eval_metric":"logloss"},X[tr],(y[tr]>0).astype(int),w,X[te])
pos = y[tr]>0
pp = pred({**base,"objective":"reg:quantileerror","quantile_alpha":0.5,"eval_metric":"quantile"},X[tr][pos],y[tr][pos],w[pos],X[te])
two = np.where(pb>0.5, 0.0, pp)
print(f"t={time.time()-t0:.0f}s quantile {np.abs(pq-yv).mean():.2f} | two-part {np.abs(two-yv).mean():.2f} | blend .5 {np.abs((0.5*pq+0.5*two)-yv).mean():.2f}")
print("P(0) mean", pb.mean().round(3), "actual zerofrac", (yv==0).mean().round(3))
# log model
yl=np.log1p(y[tr])
pl = pred({**base,"objective":"reg:squarederror","eval_metric":"mae"},X[tr],yl,w,X[te])
b2=xgb.train({**base,"objective":"reg:squarederror","seed":11},xgb.DMatrix(X[tr],label=yl,weight=w),1200)
fac=np.mean(np.exp(yl-b2.predict(xgb.DMatrix(X[tr]))))
plog=np.expm1(pl)*fac
print("log-smeared", np.abs(plog-yv).mean().round(2), "| blend q+log .7/.3", np.abs((0.7*pq+0.3*plog)-yv).mean().round(2), "| q+log+2p", np.abs((0.6*pq+0.2*plog+0.2*two)-yv).mean().round(2))
