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
base=dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,eval_metric="quantile",tree_method="hist")
def pred(ps,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    ps=dict(ps); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds); pr.append(b.predict(xgb.DMatrix(Xte)))
    return np.mean(pr,axis=0)
# ref model on 431 and on 403 (for calibration fit)
tr=day<431; te=day==431; w=0.5**((431-day[tr])/140.0)
p431 = pred(base,X[tr],y[tr],w,X[te]); yv=y[te]
tr2=day<403; te2=day==403; w2=0.5**((403-day[tr2])/140.0)
p403 = pred(base,X[tr2],y[tr2],w2,X[te2]); y403=y[te2]
print("ref 431 MAE", np.abs(p431-yv).mean().round(2), "| 403 MAE", np.abs(p403-y403).mean().round(2))
# calibration: linear y ~ p fit on day-403 preds, applied to 431 preds
b,a = np.polyfit(p403,y403,1)
print("calib slope",round(b,3),"intercept",round(a,1))
pc = b*p431+a
print("linear-calib 431 MAE", np.abs(pc-yv).mean().round(2), " mean pred", pc.mean().round(1))
# isotonic
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip").fit(p403,y403)
pi = iso.predict(p431)
print("isotonic-calib 431 MAE", np.abs(pi-yv).mean().round(2), " mean pred", pi.mean().round(1))
# regularized variant
ps20 = {**base,"min_child_weight":20}
pr20 = pred(ps20,X[tr],y[tr],w,X[te])
print("mcw20 431 MAE", np.abs(pr20-yv).mean().round(2))
