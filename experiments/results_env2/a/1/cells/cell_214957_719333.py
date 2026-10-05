
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
drop = ["index","household_key","snapshot_day","y"]
Xcols = [c for c in f.columns if c not in drop]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y

def ev(params, num=1200, logt=False):
    ytr_ = np.log1p(ytr) if logt else ytr
    m = xgb.XGBRegressor(n_estimators=num, **params)
    m.fit(Xtr, ytr_)
    p = m.predict(Xho)
    if logt: p = np.expm1(p)
    return round(np.abs(p-yho).mean(),3), m, p

base = dict(objective="reg:pseudohubererror", tree_method="hist", n_jobs=8)
for lr,depth,mcw in [(0.03,5,5),(0.02,5,10),(0.02,4,10),(0.02,6,10),(0.01,5,10)]:
    mae,_,_ = ev({**base,"learning_rate":lr,"max_depth":depth,"min_child_weight":mcw,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5})
    print("huber",lr,depth,mcw,"->",mae)
for lr,depth in [(0.03,5),(0.02,5)]:
    mae,_,_ = ev({**base,"objective":"reg:squarederror","learning_rate":lr,"max_depth":depth,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5}, logt=True)
    print("log1p-L2",lr,depth,"->",mae)
mae,m,p = ev({**base,"learning_rate":0.02,"max_depth":5,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5})
for b in [0.1,0.2,0.3,0.5]:
    print("blend b=",b, round(np.abs((b*ho.spend_84/3+(1-b)*p)-yho).mean(),3))
