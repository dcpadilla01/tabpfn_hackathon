
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]
tr = f[f.snapshot_day.isin(tr_days)].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y
def mae(p,y): return round(float(np.abs(p-y).mean()),3)
def qm(X,y,alpha=0.5,lr=0.02,depth=5,mcw=5,num=1800,sw=None):
    m = xgb.XGBRegressor(n_estimators=num, objective="reg:quantileerror", quantile_alpha=alpha,
                         learning_rate=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(X,y,sample_weight=sw); return m
# time-decay weights
w = np.exp(-(431-tr.snapshot_day.values)/200.0)
m = qm(Xtr,ytr,sw=w); print("decay200:", mae(m.predict(Xho),yho))
w2 = np.exp(-(431-tr.snapshot_day.values)/400.0)
m2 = qm(Xtr,ytr,sw=w2); print("decay400:", mae(m2.predict(Xho),yho))
# two-part: classifier y==0
clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.7,
                        reg_lambda=5, tree_method="hist", n_jobs=8, eval_metric="logloss")
clf.fit(Xtr,(ytr==0).astype(int))
pzero = clf.predict_proba(Xho)[:,1]
base = m.predict(Xho)
for t in [0.5,0.6,0.7]:
    p = np.where(pzero>t, 0.0, base)
    print("2part t=",t, mae(p,yho))
p3 = base*(1-np.minimum(pzero,0.9)); print("scale by (1-p0):", mae(p3,yho))
# feature importance top15
imp = pd.Series(m.feature_importances_, index=Xcols).sort_values(ascending=False)
print(imp.head(15).round(4).to_dict())
