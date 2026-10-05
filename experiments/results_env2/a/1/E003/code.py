
import pandas as pd, numpy as np

tt = agent_api.train_targets()
y = tt.future_spend_4w
print("train rows:", len(tt), "| zero frac:", (y==0).mean().round(3), "| mean:", y.mean().round(2), "| median:", y.median().round(2), "| p90:", y.quantile(.9).round(2), "| max:", y.max().round(1))

f = agent_api.load_saved("e002_features.parquet")
print("e002_features cols:", f.columns.tolist())
print("e002 shape:", f.shape)

days = agent_api.snapshot_days()["train"]
res = {}
for d in days:
    v = agent_api.snapshot(as_of_day=d)
    tx = v.table("transactions")
    hh = pd.Index(v.households)
    def spend(win, offset=0):
        m = (tx.day > d - offset - win) & (tx.day <= d - offset)
        return tx[m].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    s28, s56, s84, s168 = spend(28), spend(56), spend(84), spend(168)
    s_ly = spend(28, 364)  # same 4-week block one year earlier
    last = tx.groupby("household_key").day.max().reindex(hh)
    rec = (d - last).fillna(999)
    yt = tt[tt.snapshot_day==d].set_index("household_key").future_spend_4w
    idx = yt.index
    preds = {
        "s28": s28.reindex(idx),
        "s56/2": (s56/2).reindex(idx),
        "s84/3": (s84/3).reindex(idx),
        "s168/6": (s168/6).reindex(idx),
        "s_ly": s_ly.reindex(idx),
        "blend": (0.5*s28 + 0.3*s56/2 + 0.2*s84/3).reindex(idx),
    }
    for k,p in preds.items():
        res.setdefault(k, []).append((p - yt).abs().mean())
    if d == days[-1]:
        df = pd.DataFrame({"y": yt, "s28": s28.reindex(idx), "s84": s84.reindex(idx), "s_ly": s_ly.reindex(idx), "rec": rec.reindex(idx)})
        print("corr with y @day", d, ":", df.corr().y.round(3).to_dict())
        print("zero frac of y @", d, (yt==0).mean().round(3), "| zero frac of s28:", (s28.reindex(idx)==0).mean().round(3))
for k, v_ in res.items():
    print("naive MAE", k, round(float(np.mean(v_)), 3))
print("n hh per snapshot:", len(agent_api.snapshot(as_of_day=459).households))


# ---- cell ----

v = agent_api.snapshot(as_of_day=459)
print(type(v.households))
print(v.households.head() if hasattr(v.households, "head") else v.households[:5])
tx = v.table("transactions")
print(tx.shape, tx.day.min(), tx.day.max())


# ---- cell ----

import pandas as pd, numpy as np

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    def spend(win, offset=0):
        m = (tx.day > d - offset - win) & (tx.day <= d - offset)
        return tx[m].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    s28, s56, s84, s168 = spend(28), spend(56), spend(84), spend(168)
    s_ly = spend(28, 364)
    last = tx.groupby("household_key").day.max().reindex(hh)
    rec = (d - last).fillna(999)
    return pd.DataFrame({
        "s28": s28, "s56_2": s56/2, "s84_3": s84/3, "s168_6": s168/6, "s_ly": s_ly,
        "blend": 0.5*s28 + 0.3*s56/2 + 0.2*s84/3,
        "rec": rec,
    }, index=hh)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
m = feats.join(tt.future_spend_4w, on=["household_key","snapshot_day"])
print("rows:", len(m), "y nan:", m.future_spend_4w.isna().sum())
for k in ["s28","s56_2","s84_3","s168_6","s_ly","blend"]:
    print("naive MAE", k, round((m[k]-m.future_spend_4w).abs().mean(),3))
for d in sorted(m.snapshot_day.unique()):
    sub = m[m.snapshot_day==d]
    print(d, "corr:", {k: round(np.corrcoef(sub[k], sub.future_spend_4w)[0,1],3) for k in ["s28","s56_2","s84_3","s_ly","rec"]},
          "zero_y:", round((sub.future_spend_4w==0).mean(),3), "zero_s28:", round((sub.s28==0).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb

f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
train = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
hold = f[f.snapshot_day==431].dropna(subset=["y"])
print(len(train), len(hold))

drop = ["index","household_key","snapshot_day","y"]
Xcols = [c for c in f.columns if c not in drop]
Xtr, ytr = train[Xcols], train.y
Xho, yho = hold[Xcols], hold.y
print("naive s84/3 holdout MAE:", round((hold.spend_84/3 - yho).abs().mean(),3))
print("naive s28 holdout MAE:", round((hold.spend_28 - yho).abs().mean(),3))

def fit_eval(params, Xtr=Xtr, ytr=ytr, Xho=Xho, yho=yho, num=800):
    m = xgb.XGBRegressor(n_estimators=num, tree_method="hist", n_jobs=8, **params)
    m.fit(Xtr, ytr)
    p = m.predict(Xho)
    return round(np.abs(p-yho).mean(),3), m, p

for obj in ["reg:squarederror","reg:absoluteerror","reg:pseudohubererror"]:
    for lr, depth in [(0.05,6),(0.03,5)]:
        mae,_,_ = fit_eval(dict(objective=obj, learning_rate=lr, max_depth=depth, subsample=0.8, colsample_bytree=0.7, reg_lambda=5))
        print(obj, lr, depth, "MAE:", mae)


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
drop = ["index","household_key","snapshot_day","y"]
Xcols = [c for c in f.columns if c not in drop]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y

def ev(params, num=1200, logt=False, blend=None):
    ytr_ = np.log1p(ytr) if logt else ytr
    m = xgb.XGBRegressor(n_estimators=num, tree_method="hist", n_jobs=8, **params)
    m.fit(Xtr, ytr_)
    p = m.predict(Xho)
    if logt: p = np.expm1(p)
    mae = np.abs(p-yho).mean()
    if blend: p2 = blend*ho.spend_84/3 + (1-blend)*p; mae = min(mae, np.abs(p2-yho).mean())
    return round(mae,3), m

base = dict(objective="reg:pseudohubererror", tree_method="hist", n_jobs=8)
for lr,depth,mcw in [(0.03,5,5),(0.02,5,10),(0.02,4,10),(0.02,6,10),(0.01,5,10)]:
    mae,_ = ev({**base,"learning_rate":lr,"max_depth":depth,"min_child_weight":mcw,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5})
    print("huber",lr,depth,mcw,"->",mae)
# log1p target with L2
for lr,depth in [(0.03,5),(0.02,5)]:
    mae,_ = ev({**base,"objective":"reg:squarederror","learning_rate":lr,"max_depth":depth,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5}, logt=True)
    print("log1p-L2",lr,depth,"->",mae)
# blend best huber with naive
mae,m = ev({**base,"learning_rate":0.02,"max_depth":5,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.7,"reg_lambda":5})
p = m.predict(Xho)
for b in [0.1,0.2,0.3,0.5]:
    p2 = b*ho.spend_84/3 + (1-b)*p
    print("blend b=",b, round(np.abs(p2-yho).mean(),3))


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y
def mae(p): return round(float(np.abs(p-yho).mean()),3)

try:
    m = xgb.XGBRegressor(n_estimators=1200, objective="reg:quantileerror", quantile_alpha=0.5,
                         learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(Xtr, ytr); print("quantile median:", mae(m.predict(Xho)))
except Exception as e: print("quantile err:", type(e).__name__, e)

m = xgb.XGBRegressor(n_estimators=1200, objective="reg:pseudohubererror", learning_rate=0.03, max_depth=5,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
m.fit(Xtr, ytr); p = m.predict(Xho)
for q in [0.97,0.99,1.0]:
    c = np.quantile(p, q); print("clip@",q, mae(np.minimum(p,c)))
for lr,num in [(0.02,1800),(0.015,2400)]:
    m2 = xgb.XGBRegressor(n_estimators=num, objective="reg:pseudohubererror", learning_rate=lr, max_depth=5,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m2.fit(Xtr, ytr); print("huber",lr,num, mae(m2.predict(Xho)))

from sklearn.ensemble import HistGradientBoostingRegressor
hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05, min_samples_leaf=40,
                                   l2_regularization=1.0, random_state=0)
hg.fit(Xtr, ytr); print("skHGB abs:", mae(hg.predict(Xho)))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
print(f.groupby("snapshot_day").y.agg(["mean","median"]).round(1).T)
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y
def mae(p): return round(float(np.abs(p-yho).mean()),3)
def qmodel(alpha, lr=0.03, depth=5, mcw=5, num=1200):
    m = xgb.XGBRegressor(n_estimators=num, objective="reg:quantileerror", quantile_alpha=alpha,
                         learning_rate=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(Xtr, ytr); return m
preds = {}
for a in [0.45,0.5,0.55,0.6]:
    m = qmodel(a); p = m.predict(Xho); preds[a]=p; print("q",a, mae(p))
# ensemble of quantile + skHGB
from sklearn.ensemble import HistGradientBoostingRegressor
hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05, min_samples_leaf=40,
                                   l2_regularization=1.0, random_state=0)
hg.fit(Xtr, ytr); ph = hg.predict(Xho); print("skHGB:", mae(ph))
for w in [0.3,0.5,0.7]:
    print("ens q0.5+skHGB w=",w, mae(w*preds[0.5]+(1-w)*ph))
ens = np.mean([preds[a] for a in preds]); print("ens all q:", mae(ens))
# bias check by snapshot for q0.5
m50 = qmodel(0.5)
val_all = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].dropna(subset=["y"])
pv = m50.predict(val_all[Xcols])
r = pd.DataFrame({"d":val_all.snapshot_day,"p":pv,"y":val_all.y})
print(r.groupby("d").apply(lambda g: pd.Series({"bias":(g.p-g.y).mean(),"mae":(g.p-g.y).abs().mean()}), include_groups=False).round(2))


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].dropna(subset=["y"])
val = f[f.snapshot_day.isin([459,487,515,543])]
print("train rows:", len(tr), "val rows:", len(val))

w = np.exp(-(431 - tr.snapshot_day.values)/400.0)
m = xgb.XGBRegressor(n_estimators=1800, objective="reg:quantileerror", quantile_alpha=0.5,
                     learning_rate=0.02, max_depth=5, min_child_weight=5, subsample=0.8,
                     colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
m.fit(tr[Xcols], tr.y, sample_weight=w)
clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5, subsample=0.8,
                        colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8, eval_metric="logloss")
clf.fit(tr[Xcols], (tr.y==0).astype(int))
p = m.predict(val[Xcols])
p0 = clf.predict_proba(val[Xcols])[:,1]
p = p * (1 - np.minimum(p0, 0.9))
p = np.clip(p, 0, None)
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = p
print("pred stats: mean", round(p.mean(),1), "median", round(np.median(p),1), "zeros", round((p<1).mean(),3), "max", round(p.max(),1))
path = agent_api.save_table(out, "e003_preds.parquet")
print(path)
