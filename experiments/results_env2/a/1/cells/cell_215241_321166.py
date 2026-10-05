
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]
tr = f[f.snapshot_day.isin(tr_days)].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
def qmodel(X,y,alpha=0.5,lr=0.03,depth=5,mcw=5,num=1200,cols=None):
    m = xgb.XGBRegressor(n_estimators=num, objective="reg:quantileerror", quantile_alpha=alpha,
                         learning_rate=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(X if cols is None else X[cols], y); return m
def mae(p,y): return round(float(np.abs(p-y).mean()),3)

m = qmodel(tr[Xcols], tr.y)
# median residual on train snapshots (in-sample) and holdout
ptr = m.predict(tr[Xcols]); pho = m.predict(ho[Xcols])
print("train med resid:", round(float(np.median(ptr-tr.y)),2), "| holdout med resid:", round(float(np.median(pho-ho.y)),2))
# optimal shift estimated on train, applied to holdout
c = float(np.median(tr.y - ptr))
print("shift c:", round(c,2), "-> holdout MAE:", mae(pho+c, ho.y))
# multiplicative calib
g = float(np.sum(tr.y*ptr)/np.sum(ptr*ptr))
print("mult g:", round(g,3), "-> holdout MAE:", mae(g*pho, ho.y))
# drop calendar features
cal = ["week_of_year","week_sin","week_cos"]
m2 = qmodel(tr[Xcols], tr.y); p2 = m2.predict(ho[Xcols]); print("with cal:", mae(p2,ho.y))
m3 = qmodel(tr[[c for c in Xcols if c not in cal]], tr.y); print("no cal:", mae(m3.predict(ho[[c for c in Xcols if c not in cal]]), ho.y))
# depth 7 / lr 0.02 num 1800
m4 = qmodel(tr[Xcols], tr.y, depth=7); print("depth7:", mae(m4.predict(ho[Xcols]), ho.y))
m5 = qmodel(tr[Xcols], tr.y, lr=0.02, num=1800); print("lr.02/1800:", mae(m5.predict(ho[Xcols]), ho.y))
# two-stage: predict y and also log1p(y) median, average
m6 = qmodel(tr[Xcols], np.log1p(tr.y)); p6 = np.expm1(m6.predict(ho[Xcols])); print("log-q:", mae(p6, ho.y))
print("avg q+logq:", mae(0.5*(pho+p6), ho.y))
