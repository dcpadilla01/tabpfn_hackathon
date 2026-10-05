import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, sklearn
print("xgb", xgb.__version__, "sklearn", sklearn.__version__)
print(A.snapshot_days())
f4 = A.load_saved("feats_v3.parquet")
print(f4.shape); print(f4.columns.tolist()[:40])
t = A.train_targets(); print(t.shape); print(t.future_spend_4w.describe())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
tr = v.transactions
print(tr.shape)
print(tr.groupby('household_key').day.agg(['min','max','count']).describe())
# gap structure: days between baskets per household
g = tr.sort_values(['household_key','day']).groupby('household_key').day.apply(lambda s: s.drop_duplicates().diff().dropna())
print(g.describe())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f4 = A.load_saved("feats_v3.parquet")
print(f4.columns.tolist()[40:])
t = A.train_targets()
m = f4.merge(t, on=['household_key','snapshot_day'])
print(m.shape)
num = m.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','index'], errors='ignore')
cor = num.corr()['future_spend_4w'].sort_values(key=abs, ascending=False)
print(cor.head(25).round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
va_days = [459,487,515,543]

Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w
Xva = f[f.snapshot_day.isin(va_days)]
print(Xtr.shape, Xes.shape, Xva.shape)

def fit_predict(objective, ytr_raw, params=None, n=1200, lr=0.03):
    p = dict(n_estimators=n, learning_rate=lr, max_depth=7, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective=objective,
             tree_method='hist', n_jobs=4)
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr_raw, eval_set=[(Xes, yes)], verbose=False)
    return m

# A: plain squared on raw (E004-style reference, quick check)
mA = fit_predict('reg:squarederror', ytr)
# B: squared on log1p
mB = fit_predict('reg:squarederror', np.log1p(ytr))
# C: pseudo-huber raw
mC = fit_predict('reg:pseudohubererror', ytr)

va_pred_A = np.clip(mA.predict(Xva[feat_cols]), 0, None)
va_pred_B = np.clip(np.expm1(mB.predict(Xva[feat_cols])), 0, None)
va_pred_C = np.clip(mC.predict(Xva[feat_cols]), 0, None)

# check on ES snapshots (375/403/431) as a proxy for val
esA = np.clip(mA.predict(Xes), 0, None); esB = np.clip(np.expm1(mB.predict(Xes)),0,None); esC = np.clip(mC.predict(Xes),0,None)
print("ES MAE  raw:", np.abs(esA-yes).mean().round(3), " log:", np.abs(esB-yes).mean().round(3), " huber:", np.abs(esC-yes).mean().round(3))
print("ES R2   raw:", 1-((esA-yes)**2).sum()/((yes-yes.mean())**2).sum(), " log:", 1-((esB-yes)**2).sum()/((yes-yes.mean())**2).sum(), " huber:", 1-((esC-yes)**2).sum()/((yes-yes.mean())**2).sum())
for w in [0.0,0.25,0.5,0.75,1.0]:
    print("blend w_log=",w, "ES MAE", np.abs(w*esB+(1-w)*esA-yes).mean().round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, obj, n, lr, md, mcw, subs=0.8, col=0.7, lam=1.0):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=subs, colsample_bytree=col, reg_lambda=lam, objective=obj,
        tree_method='hist', n_jobs=4)
    m.fit(Xtr, ytr, verbose=False)
    p = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p-yes).mean(),3), "R2", round(1-((p-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))

run("huber n1200 lr.03 d7", 'reg:pseudohubererror', 1200, .03, 7, 5)
run("huber n2500 lr.015 d7", 'reg:pseudohubererror', 2500, .015, 7, 5)
run("huber n2500 lr.015 d6", 'reg:pseudohubererror', 2500, .015, 6, 5)
run("huber n2500 lr.015 d8", 'reg:pseudohubererror', 2500, .015, 8, 5)
run("huber n2500 lr.015 d7 mcw20", 'reg:pseudohubererror', 2500, .015, 7, 20)
run("quant.5 n2500 lr.015 d7", 'reg:quantileerror', 2500, .015, 7, 5, params=None) if False else None
run("quant.5 n2500 lr.015 d7", 'reg:quantileerror', 2500, .015, 7, 5)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, obj, n, lr, md, mcw, extra=None, ret=False):
    p = dict(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective=obj,
        tree_method='hist', n_jobs=4)
    if extra: p.update(extra)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr, verbose=False)
    p_ = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p_-yes).mean(),3), "R2", round(1-((p_-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))
    return p_

p_hub = run("huber base", 'reg:pseudohubererror', 2500, .015, 7, 5)
p_q50 = run("quantile .5", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.5})
p_q45 = run("quantile .45", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.45})
p_q55 = run("quantile .55", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.55})
for w in [0.25,0.5,0.75]:
    bl = w*p_hub + (1-w)*p_q50
    print("blend hub/q50 w_hub=",w, "ES MAE", round(np.abs(bl-yes).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, n, lr):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective='reg:quantileerror',
        quantile_alpha=0.5, tree_method='hist', n_jobs=4)
    m.fit(Xtr, ytr, verbose=False)
    p_ = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p_-yes).mean(),3), "R2", round(1-((p_-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))

run("q50 n1200 lr.03 (E004-matched)", 1200, .03)
run("q50 n2500 lr.015", 2500, .015)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
print(sorted(f.snapshot_day.unique()))
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
va_days = [459,487,515,543]

Xtr, ytr = df[feat_cols], df.future_spend_4w.values
Xva = f[f.snapshot_day.isin(va_days)]
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective='reg:quantileerror',
    quantile_alpha=0.5, tree_method='hist', n_jobs=4)
m.fit(Xtr, ytr, verbose=False)
pred = np.clip(m.predict(Xva[feat_cols]), 0, None)
out = Xva[['household_key','snapshot_day']].copy()
out['prediction'] = pred
print(out.shape, out.snapshot_day.value_counts().to_dict())
p = A.save_table(out, "pred_e005.parquet")
print(p)
