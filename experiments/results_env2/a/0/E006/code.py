import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("feats_v3.parquet")
print("feats_v3:", feats.shape)
print(feats.columns.tolist())
print(feats.head(3).T)
pred = agent_api.load_saved("pred_e005.parquet")
print("\npred_e005:", pred.shape)
print(pred.head())
print("\nsnapshot days:", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("train_targets:", tt.shape)
print(tt.describe())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
CAT = ["classification_1","classification_3","classification_4","classification_5","classification_2","homeowner","kids"]
for c in CAT: df[c] = df[c].astype("category")
tr = df[df.snapshot_day <= 431].copy()
va = df[df.snapshot_day >= 459].copy()
print("train rows", len(tr), "val rows", len(va), "nan cols:", [c for c in FE if tr[c].isna().any()])

PARAMS = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
              tree_method="hist", enable_categorical=True, n_jobs=4)

# reproduce E005: quantile 0.5 on full train
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m.fit(tr[FE], tr.future_spend_4w)
p = m.predict(va[FE])
old = agent_api.load_saved("pred_e005.parquet").sort_values(["household_key","snapshot_day"]).prediction.values
va_sorted = va.sort_values(["household_key","snapshot_day"])
print("MAE repro:", mean_absolute_error(va_sorted.future_spend_4w, p), " vs old:", mean_absolute_error(va_sorted.future_spend_4w, old), " corr:", np.corrcoef(p, old)[0,1])


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
tr = df[df.snapshot_day <= 431].copy()
va = df[df.snapshot_day >= 459].copy()
print("train", len(tr), "val", len(va))

PARAMS = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              tree_method="hist", n_jobs=4)
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m.fit(tr[FE], tr.future_spend_4w)
p = m.predict(va[FE])
va_sorted = va.sort_values(["household_key","snapshot_day"]).reset_index(drop=True)
old = agent_api.load_saved("pred_e005.parquet").sort_values(["household_key","snapshot_day"]).prediction.values
print("MAE repro:", mean_absolute_error(va_sorted.future_spend_4w, p), "old:", mean_absolute_error(va_sorted.future_spend_4w, old), "corr:", np.corrcoef(p, old)[0,1])


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
tr = df[df.snapshot_day <= 431].copy()
va = df[df.snapshot_day >= 459].copy()

PARAMS = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              tree_method="hist", n_jobs=4)
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m.fit(tr[FE], tr.future_spend_4w)
p = m.predict(va[FE])
old = agent_api.load_saved("pred_e005.parquet").sort_values(["household_key","snapshot_day"]).prediction.values
print("corr with old E005 preds:", np.corrcoef(p, old)[0,1], " mean|diff|:", np.abs(p-old).mean())

# target stats by snapshot day (train only)
print(df.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))

# local pseudo-val: train on <=403, eval on 431
tr2 = df[df.snapshot_day <= 403]
pv = df[df.snapshot_day == 431]
m2 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m2.fit(tr2[FE], tr2.future_spend_4w)
pp = m2.predict(pv[FE])
print("pseudo-val 431 MAE:", mean_absolute_error(pv.future_spend_4w, pp))
# naive baselines on 431
print("lag0 (spend_28) MAE:", mean_absolute_error(pv.future_spend_4w, pv.spend_28))
print("exp4w_blend MAE:", mean_absolute_error(pv.future_spend_4w, pv.exp4w_blend))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)

print("zero frac in train targets:", (df.future_spend_4w==0).mean())

BASE = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=4)
def fit_eval(cfg, train_max, eval_day, obj="reg:quantileerror", q=0.5):
    tr = df[df.snapshot_day <= train_max]; pv = df[df.snapshot_day == eval_day]
    p = dict(BASE); p.update(cfg)
    m = xgb.XGBRegressor(objective=obj, quantile_alpha=q, **p)
    m.fit(tr[FE], tr.future_spend_4w)
    pr = np.clip(m.predict(pv[FE]), 0, None)
    return mean_absolute_error(pv.future_spend_4w, pr), pr

t0=time.time()
# E005 config as reference on 431
mae0, pr0 = fit_eval(dict(n_estimators=900, max_depth=6, min_child_weight=5), 403, 431)
print(f"E005 cfg on 431: {mae0:.3f}  min pred {pr0.min():.2f}  ({time.time()-t0:.0f}s)")

# bias check per day (rolling origin)
for d in [347, 375, 403, 431]:
    mae, pr = fit_eval(dict(n_estimators=900, max_depth=6, min_child_weight=5), d-28, d)
    act = df[df.snapshot_day==d].future_spend_4w
    print(f"day {d}: MAE {mae:.2f}  mean pred {pr.mean():.1f} vs mean act {act.mean():.1f}  median pred {np.median(pr):.1f} vs {act.median():.1f}")


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)

DAYS = [375, 403, 431]
def preds(obj, q, train_max):
    tr = df[df.snapshot_day <= train_max]
    m = xgb.XGBRegressor(objective=obj, quantile_alpha=q, **BASE)
    m.fit(tr[FE], tr.future_spend_4w)
    return {d: np.clip(m.predict(df[df.snapshot_day==d][FE]),0,None) for d in DAYS}

t0=time.time()
P_q50 = {}; P_q60 = {}; P_hub = {}
for d in DAYS:
    P_q50.update(preds("reg:quantileerror", 0.5, d-28))
    P_q60.update(preds("reg:quantileerror", 0.6, d-28))
    P_hub.update(preds("reg:pseudohubererror", 0.5, d-28))
print(f"fit time {time.time()-t0:.0f}s")

act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}
def ev(getter, label):
    maes = [mean_absolute_error(act[d], getter(d)) for d in DAYS]
    print(f"{label:28s} MAEs {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev(lambda d: P_q50[d], "q50")
for s in [1.05, 1.10, 1.15, 1.20]:
    ev(lambda d, s=s: P_q50[d]*s, f"q50 x{s}")
ev(lambda d: P_q60[d], "q60")
for w in [0.3, 0.5, 0.7]:
    ev(lambda d, w=w: w*P_q50[d] + (1-w)*P_hub[d], f"blend q50+hub w={w}")
ev(lambda d: P_hub[d], "huber alone")


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)

tr = df[df.snapshot_day <= 403]; pv = df[df.snapshot_day == 431].copy()
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE)
m.fit(tr[FE], tr.future_spend_4w)
pv["pred"] = np.clip(m.predict(pv[FE]),0,None)

# error decomposition by actual bucket
pv["bucket"] = pd.cut(pv.future_spend_4w, [-1, 0.01, 25, 75, 150, 300, 10000])
g = pv.groupby("bucket", observed=True).apply(lambda x: pd.Series({
    "n": len(x), "mean_pred": x.pred.mean(), "mean_act": x.future_spend_4w.mean(),
    "MAE": np.abs(x.pred - x.future_spend_4w).mean(), "err_share": np.abs(x.pred-x.future_spend_4w).sum()/np.abs(pv.pred-pv.future_spend_4w).sum()}))
print(g)

# zero-spend rows: what does the model say?
z = pv[pv.future_spend_4w == 0]
print("\nzero rows:", len(z), "mean pred:", z.pred.mean().round(2), "median pred:", z.pred.median().round(2))
print("their exp4w_blend:", z.exp4w_blend.mean().round(2), "spend_28:", z.spend_28.mean().round(2), "days_since_last:", z.days_since_last.mean().round(1))

# how predictive is a zero-classifier signal? fraction of 4w windows with zero spend by recent activity
df["zero_next"] = (df.future_spend_4w == 0).astype(float)
for c in ["days_since_last","spend_28","wk0","active_weeks8"]:
    print(c, "corr with zero_next:", df[[c,"zero_next"]].corr().iloc[0,1].round(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)
DAYS = [375, 403, 431]
act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}

def run(train_max):
    tr = df[df.snapshot_day <= train_max]
    nz = tr[tr.future_spend_4w > 0]
    mq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mq.fit(tr[FE], tr.future_spend_4w)
    mc = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4,
                           eval_metric="logloss")
    mc.fit(tr[FE], (tr.future_spend_4w == 0).astype(int))
    mn = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mn.fit(nz[FE], nz.future_spend_4w)
    ml = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); ml.fit(tr[FE], np.log1p(tr.future_spend_4w))
    out = {}
    for d in DAYS:
        X = df[df.snapshot_day==d]
        pq = np.clip(mq.predict(X[FE]),0,None)
        pz = mc.predict_proba(X[FE])[:,1]
        pn = np.clip(mn.predict(X[FE]),0,None)
        pl = np.clip(np.expm1(ml.predict(X[FE])),0,None)
        out[d] = dict(q50=pq.values, pz=pz.values, nz=pn.values,
                      twopart=(1-pz)*pn, twothr=np.where(pz>0.5, 0, pn), log=pl.values)
    return out

t0=time.time()
R = {d: run(d-28) for d in DAYS}
print(f"fits done {time.time()-t0:.0f}s")
def ev(key, label, post=None):
    maes=[]
    for d in DAYS:
        p = R[d][key].copy()
        if post: p = post(p)
        maes.append(mean_absolute_error(act[d], p))
    print(f"{label:24s} {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev("q50","q50")
ev("log","log-target q50")
ev("nz","nz-only q50")
ev("twopart","twopart (1-pz)*nz")
ev("twothr","twopart thr .5")
for t in [0.3,0.7]:
    ev("nz", f"nz thr {t}", post=lambda p,t=t: np.where(R[DAYS[0]] and p, p, p))  # placeholder


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)
DAYS = [375, 403, 431]
act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}

def run(train_max):
    tr = df[df.snapshot_day <= train_max]
    nz = tr[tr.future_spend_4w > 0]
    mq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mq.fit(tr[FE], tr.future_spend_4w)
    mc = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4,
                           eval_metric="logloss")
    mc.fit(tr[FE], (tr.future_spend_4w == 0).astype(int))
    mn = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mn.fit(nz[FE], nz.future_spend_4w)
    ml = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); ml.fit(tr[FE], np.log1p(tr.future_spend_4w))
    out = {}
    for d in DAYS:
        X = df[df.snapshot_day==d]
        pq = np.clip(mq.predict(X[FE]),0,None)
        pz = mc.predict_proba(X[FE])[:,1]
        pn = np.clip(mn.predict(X[FE]),0,None)
        pl = np.clip(np.expm1(ml.predict(X[FE])),0,None)
        out[d] = dict(q50=pq, pz=pz, nz=pn, twopart=(1-pz)*pn, twothr=np.where(pz>0.5, 0, pn), log=pl)
    return out

t0=time.time()
R = {d: run(d-28) for d in DAYS}
print(f"fits done {time.time()-t0:.0f}s")
def ev(key, label):
    maes=[mean_absolute_error(act[d], R[d][key]) for d in DAYS]
    print(f"{label:24s} {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev("q50","q50"); ev("log","log-target q50"); ev("nz","nz-only q50")
ev("twopart","twopart (1-pz)*nz"); ev("twothr","twopart thr .5")
# two-part variants with soft scaling
for k in [0.5, 0.8]:
    ev("nz", f"nz*(1-pz*{k})", post=None)
    for d in DAYS: R[d][f"nz{k}"] = (1-k*R[d]["pz"])*R[d]["nz"]
    ev(f"nz{k}", f"nz*(1-{k}*pz)")
# blend q50 with twothr
for w in [0.3, 0.5]:
    for d in DAYS: R[d][f"bl{w}"] = w*R[d]["q50"] + (1-w)*R[d]["twothr"]
    ev(f"bl{w}", f"blend q50+twothr w={w}")


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)
DAYS = [375, 403, 431]
act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}

def run(train_max, eval_day):
    tr = df[df.snapshot_day <= train_max]
    nz = tr[tr.future_spend_4w > 0]
    mq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mq.fit(tr[FE], tr.future_spend_4w)
    mc = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4,
                           eval_metric="logloss")
    mc.fit(tr[FE], (tr.future_spend_4w == 0).astype(int))
    mn = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mn.fit(nz[FE], nz.future_spend_4w)
    ml = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); ml.fit(tr[FE], np.log1p(tr.future_spend_4w))
    X = df[df.snapshot_day==eval_day]
    pq = np.clip(mq.predict(X[FE]),0,None)
    pz = mc.predict_proba(X[FE])[:,1]
    pn = np.clip(mn.predict(X[FE]),0,None)
    pl = np.clip(np.expm1(ml.predict(X[FE])),0,None)
    return dict(q50=pq, pz=pz, nz=pn, twopart=(1-pz)*pn, twothr=np.where(pz>0.5,0,pn), log=pl)

t0=time.time()
R = {d: run(d-28, d) for d in DAYS}
print(f"fits done {time.time()-t0:.0f}s")
def ev(key, label):
    maes=[mean_absolute_error(act[d], R[d][key]) for d in DAYS]
    print(f"{label:26s} {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev("q50","q50"); ev("log","log-target q50"); ev("nz","nz-only q50")
ev("twopart","twopart (1-pz)*nz"); ev("twothr","twopart thr .5")
for k in [0.5, 0.8]:
    for d in DAYS: R[d][f"nz{k}"] = (1-k*R[d]["pz"])*R[d]["nz"]
    ev(f"nz{k}", f"nz*(1-{k}*pz)")
for w in [0.3, 0.5]:
    for d in DAYS: R[d][f"bl{w}"] = w*R[d]["q50"] + (1-w)*R[d]["twothr"]
    ev(f"bl{w}", f"blend q50+twothr w={w}")
