import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
def train_ps(Xtr,ytr,wtr,Xte,ps,rounds,seed=11):
    ps=dict(ps); ps["tree_method"]="hist"; ps["seed"]=seed
    b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds)
    return b.predict(xgb.DMatrix(Xte))
hold=431; tr=day<hold; te=day==hold; yv=y[te]
w=0.5**((hold-day[tr])/140.0)
base=dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,eval_metric="quantile")
configs = {
 "d6 (ref)": {},
 "d8": dict(max_depth=8),
 "mcw20": dict(min_child_weight=20),
 "col0.5": dict(colsample_bytree=0.5),
 "lam5": dict(reg_lambda=5.0),
 "lr.05": dict(learning_rate=0.05),
 "sub.9": dict(subsample=0.9),
}
for name,ov in configs.items():
    ps={**base,**ov}
    p=np.mean([train_ps(X[tr],y[tr],w,X[te],ps,1200,s) for s in (11,22,33)],axis=0)
    print(f"{name:10s} MAE {np.abs(p-yv).mean():.2f}")
# log-target
yl=np.log1p(y[tr])
psl=dict(objective="reg:squarederror",learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",seed=11)
b=xgb.train(psl,xgb.DMatrix(X[tr],label=yl,weight=w),1200)
intr=b.predict(xgb.DMatrix(X[tr])); fac=np.mean(np.exp(yl-intr))
pl=b.predict(xgb.DMatrix(X[te])); plog=np.expm1(pl)*fac
print("log-smeared MAE", np.abs(plog-yv).mean().round(2))
