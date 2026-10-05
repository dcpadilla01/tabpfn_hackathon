

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, KEYS, TARGET, snapshot_days

feats = load_saved('feats_v3.parquet')
print('feats shape:', feats.shape)
print('cols:', list(feats.columns)[:120])
tt = train_targets()
print('targets shape:', tt.shape)
y = tt[TARGET]
print(y.describe())
print('zero rate:', round(float((y==0).mean()), 3))
print(tt.groupby('snapshot_day')[TARGET].agg(['mean','count']).round(1))


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
p5 = load_saved('pred_e005.parquet')
print('pred_e005 shape', p5.shape, p5.columns.tolist())
print(p5.groupby('snapshot_day')['prediction'].agg(['mean','median','min','max']).round(1))
# train in-sample performance per day
pv = p5.merge(tt, on=KEYS)
pv['abs_err'] = (pv['prediction']-pv[TARGET]).abs()
print(pv.groupby('snapshot_day')[['abs_err']].mean().round(2))
# correlation of prediction with target
print('corr pred-target (train):', round(pv['prediction'].corr(pv[TARGET]),3))
# error decomposition: bias by target quantile
pv['tq'] = pd.qcut(pv[TARGET], 10, duplicates='drop')
print(pv.groupby('tq').agg(n=('abs_err','size'), mae=('abs_err','mean'), bias=('prediction', lambda s: None)).round(1))
pv['bias'] = pv['prediction'] - pv[TARGET]
print(pv.groupby('tq')['bias'].mean().round(1))
print(pv.groupby('tq')[TARGET].mean().round(1), pv.groupby('tq')['prediction'].mean().round(1))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
obj_cols = [c for c in df.columns if df[c].dtype == object]
for c in obj_cols:
    df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)

tr = df[df.snapshot_day <= 403]
va = df[df.snapshot_day == 431]

def fit_med(seed, n=1200, lr=0.03, md=7, mcw=10):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror',
                         quantile_alpha=0.5, random_state=seed, n_jobs=8, tree_method='hist')
    m.fit(tr[FEATS], tr[TARGET])
    return m

preds = []
for s in [1,2,3]:
    m = fit_med(s)
    preds.append(m.predict(va[FEATS]))
p = np.mean(preds, axis=0)
mae = np.abs(p - va[TARGET]).mean()
print('E005-like MAE on day 431:', round(mae,3))

err = p - va[TARGET].values
ae = np.abs(err)
va2 = va.copy()
va2['pred'] = p; va2['err'] = err; va2['ae'] = ae
va2['tq'] = pd.qcut(va2[TARGET], 10, duplicates='drop')
g = va2.groupby('tq', observed=True).agg(n=('ae','size'), y_mean=(TARGET,'mean'), p_mean=('pred','mean'), bias=('err','mean'), mae=('ae','mean'))
print(g.round(1))
# zero spenders
z = va2[va2[TARGET]==0]
print('zero-spend rows:', len(z), 'mean pred on them:', round(z['pred'].mean(),1), 'share of total MAE:', round(z['ae'].sum()/ae.sum(),3))
nz = va2[va2[TARGET]>0]
print('nonzero MAE:', round(nz['ae'].mean(),2))
# top decile contribution
top = va2[va2[TARGET] >= va2[TARGET].quantile(0.9)]
print('top-decile rows:', len(top), 'share of MAE:', round(top['ae'].sum()/ae.sum(),3), 'their MAE:', round(top['ae'].mean(),1))
# feature importance
m0 = fit_med(1)
imp = pd.Series(m0.feature_importances_, index=FEATS).sort_values(ascending=False)
print(imp.head(25).round(4))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.isotonic import IsotonicRegression
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
tr = df[df.snapshot_day <= 403]; va = df[df.snapshot_day == 431]
yv = va[TARGET].values

def fit_med(seed, X, y, n=1200, lr=0.03, md=7, mcw=10):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        random_state=seed, n_jobs=8, tree_method='hist')
    m.fit(X, y); return m

t0=time.time()
pa = np.mean([fit_med(s, tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2,3]], axis=0)
print('fit time', round(time.time()-t0,1), 'base MAE 431:', round(np.abs(pa-yv).mean(),3))

# honest cross-fit of isotonic median calibration on day 431
rng = np.random.RandomState(0); idx = rng.permutation(len(va))
h1, h2 = idx[:len(idx)//2], idx[len(idx)//2:]
def calibrate(ptrain, ytrain, ptest):
    iso = IsotonicRegression(y_min=0, out_of_bounds='clip')
    iso.fit(ptrain, ytrain)
    return iso.predict(ptest)
pc2 = calibrate(pa[h1], yv[h1], pa[h2])
print('cross-fit recal MAE (half2): base', round(np.abs(pa[h2]-yv[h2]).mean(),3),
      '-> recal', round(np.abs(pc2-yv[h2]).mean(),3))
# bias after recal
print('half2 bias base', round((pa[h2]-yv[h2]).mean(),1), 'recal', round((pc2-yv[h2]).mean(),1))

# full calibration curve from day 431 (for later use on validation)
iso_full = IsotonicRegression(y_min=0, out_of_bounds='clip'); iso_full.fit(pa, yv)
print('cal curve samples:', [(round(x,0), round(float(iso_full.predict([x])[0]),0)) for x in [0,10,25,50,100,200,400,800,1100]])

# zero classifier
from xgboost import XGBClassifier
yz = (tr[TARGET]==0).astype(int)
clf = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, random_state=1, n_jobs=8, tree_method='hist',
                    eval_metric='logloss')
clf.fit(tr[FEATS], yz)
p0 = clf.predict_proba(va[FEATS])[:,1]
from sklearn.metrics import roc_auc_score
print('zero-classifier AUC 431:', round(roc_auc_score(yz.values if hasattr(yz,'values') else yz, p0),3))
print('base rate zero:', round(yz.mean(),3), 'mean p0 on 431:', round(p0.mean(),3))
for thr in [0.4,0.5,0.6]:
    pf = np.where(p0>thr, 0.0, pa)
    print(f'zero-override thr={thr}: MAE', round(np.abs(pf-yv).mean(),3))
# both combined (recal trained on full 431 = optimistic for 431, just indicative)
pc_all = iso_full.predict(pa)
pf2 = np.where(p0>0.5, 0.0, pc_all)
print('recal+override (in-sample cal, indicative):', round(np.abs(pf2-yv).mean(),3))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.isotonic import IsotonicRegression
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
TRAIN_DAYS = sorted(tt.snapshot_day.unique())

def fit_med(seed, X, y):
    return xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        random_state=seed, n_jobs=8, tree_method='hist').fit(X, y)

t0=time.time()
oof = []
for d in TRAIN_DAYS:
    trd = df[df.snapshot_day != d]
    vad = df[df.snapshot_day == d]
    p = np.mean([fit_med(s, trd[FEATS], trd[TARGET]).predict(vad[FEATS]) for s in [1,2]], axis=0)
    oof.append(pd.DataFrame({'household_key': vad.household_key.values, 'snapshot_day': d,
                             'pred': p, 'y': vad[TARGET].values}))
oof = pd.concat(oof, ignore_index=True)
print('OOF built in', round(time.time()-t0,1), 's; OOF MAE:', round(np.abs(oof.pred-oof.y).mean(),3))
oof['tq'] = pd.qcut(oof.y, 10, duplicates='drop')
oof['bias'] = oof.pred - oof.y
print(oof.groupby('tq', observed=True)[['y','pred','bias']].mean().round(1))
iso = IsotonicRegression(y_min=0, out_of_bounds='clip'); iso.fit(oof.pred.values, oof.y.values)
oof['pc'] = iso.predict(oof.pred.values)
print('OOF MAE after pooled recal:', round(np.abs(oof.pc-oof.y).mean(),3))
oof['tq2'] = pd.qcut(oof.pred, 10, duplicates='drop')
print(oof.groupby('tq2', observed=True)[['y','pred']].mean().round(1))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.isotonic import IsotonicRegression
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)

def fit_med(seed, X, y, n=1200):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        random_state=seed, n_jobs=8, tree_method='hist').fit(X, y)

t0=time.time()
oof = []
for d in [263, 347, 431]:
    trd = df[df.snapshot_day != d]
    vad = df[df.snapshot_day == d]
    p = fit_med(1, trd[FEATS], trd[TARGET]).predict(vad[FEATS])
    oof.append(pd.DataFrame({'snapshot_day': d, 'pred': p, 'y': vad[TARGET].values}))
oof = pd.concat(oof, ignore_index=True)
print('OOF built in', round(time.time()-t0,1), 's; OOF MAE:', round(np.abs(oof.pred-oof.y).mean(),3))
for d in [263,347,431]:
    o = oof[oof.snapshot_day==d]
    o['tq'] = pd.qcut(o.y, 10, duplicates='drop')
    b = o.groupby('tq', observed=True)[['y','pred']].mean().round(0)
    print('day', d, list(zip(b.y.astype(int), b.pred.astype(int))))
oof['tq'] = pd.qcut(oof.y, 10, duplicates='drop')
print('pooled:'); print(oof.groupby('tq', observed=True)[['y','pred']].mean().round(0))
iso = IsotonicRegression(y_min=0, out_of_bounds='clip'); iso.fit(oof.pred.values, oof.y.values)
oof['pc'] = iso.predict(oof.pred.values)
print('OOF MAE after pooled recal:', round(np.abs(oof.pc-oof.y).mean(),3), 'vs base', round(np.abs(oof.pred-oof.y).mean(),3))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.metrics import roc_auc_score
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
tr = df[df.snapshot_day <= 403]; va = df[df.snapshot_day == 431]
yv = va[TARGET].values

def fit_med(seed, X, y, qa=0.5, n=1200):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=qa,
        random_state=seed, n_jobs=8, tree_method='hist').fit(X, y)

t0=time.time()
pa = np.mean([fit_med(s, tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
print('base MAE 431:', round(np.abs(pa-yv).mean(),3), 'time', round(time.time()-t0,1))

# 1) median calibration: fraction y < pred by pred decile
q = pd.qcut(pa, 10, duplicates='drop')
frac = pd.DataFrame({'q':q, 'y':yv, 'p':pa}).groupby('q', observed=True).apply(lambda g: (g.y<g.p).mean(), include_groups=False)
print('frac(y<pred) by pred decile (0.5 = perfect median):')
print(frac.round(2))

# 2) zero classifier
yz_tr = (tr[TARGET]==0).astype(int)
clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, random_state=1, n_jobs=8, tree_method='hist', eval_metric='logloss')
clf.fit(tr[FEATS], yz_tr)
p0 = clf.predict_proba(va[FEATS])[:,1]
yz_va = (va[TARGET]==0).astype(int)
print('zero-clf AUC 431:', round(roc_auc_score(yz_va, p0),3), 'base rate:', round(yz_va.mean(),3))
for thr in [0.3,0.4,0.5,0.6]:
    pf = np.where(p0>thr, 0.0, pa)
    print(f'override thr={thr}: MAE', round(np.abs(pf-yv).mean(),3), 'zeroed rows:', int((p0>thr).sum()))

# 3) quantile_alpha sweep (single seed, cheap-ish)
for qa in [0.45, 0.55]:
    p = fit_med(1, tr[FEATS], tr[TARGET], qa=qa).predict(va[FEATS])
    print(f'qa={qa}: MAE', round(np.abs(p-yv).mean(),3))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
tr = df[df.snapshot_day <= 403]; va = df[df.snapshot_day == 431]
yv = va[TARGET].values

def base_params(obj_kwargs, seed):
    d = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=8, tree_method='hist')
    d.update(obj_kwargs); return d

t0=time.time()
# A: median ensemble (E005-like, 2 seeds)
pA = np.mean([xgb.XGBRegressor(**base_params(dict(objective='reg:quantileerror', quantile_alpha=0.5), s)
              ).fit(tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
# B: log-space L2
mB = xgb.XGBRegressor(**base_params(dict(objective='reg:squarederror'), 1))
mB.fit(tr[FEATS], np.log1p(tr[TARGET]))
pB = np.expm1(mB.predict(va[FEATS]))
# C: absoluteerror objective
pC = np.mean([xgb.XGBRegressor(**base_params(dict(objective='reg:absoluteerror'), s)
              ).fit(tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
print('time', round(time.time()-t0,1))
print('A median-ens :', round(np.abs(pA-yv).mean(),3))
print('B log-L2     :', round(np.abs(pB-yv).mean(),3))
print('C abserror   :', round(np.abs(pC-yv).mean(),3))
for w in [0.3,0.5,0.7]:
    print(f'blend A+B w={w}:', round(np.abs(w*pA+(1-w)*pB-yv).mean(),3))
print('A+B+C equal:', round(np.abs((pA+pB+pC)/3-yv).mean(),3))
# bias by decile for B
q = pd.qcut(yv, 8, duplicates='drop')
tmp = pd.DataFrame({'q':q,'y':yv,'pA':pA,'pB':pB})
print(tmp.groupby('q', observed=True)[['y','pA','pB']].mean().round(0))


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import snapshot, train_targets, KEYS, TARGET

tr = snapshot().transactions
print('transactions rows:', len(tr), 'max day:', tr.day.max(), 'n hh:', tr.household_key.nunique())
dm = snapshot().display_mailer
print('display_mailer rows:', len(dm))

# verify label computation: future_spend_4w = sum sales on days d+1..d+28
tt = train_targets()
spend = tr.groupby(['household_key','day']).sales_value.sum().reset_index()
def lab(d):
    w = spend[(spend.day > d) & (spend.day <= d+28)]
    return w.groupby('household_key').sales_value.sum()
ok, tot = 0, 0
for d in [95, 235, 431]:
    L = lab(d).rename('mylab')
    sub = tt[tt.snapshot_day==d].merge(L, on='household_key', how='left').fillna({'mylab':0})
    diff = (sub[TARGET] - sub.mylab).abs()
    print('day', d, 'n:', len(sub), 'max abs diff:', round(diff.max(),4), 'mismatch:', int((diff>0.01).sum()))
    ok += int((diff<=0.01).sum()); tot += len(sub)
print('verified rows:', ok, '/', tot)


# ---- cell ----
import pandas as pd, numpy as np, time
from agent_api import snapshot, train_targets, load_saved, KEYS, TARGET

s459 = snapshot()
prod = s459.products
print('products', prod.shape)
print(prod.department.value_counts().head(15))
print(prod.brand.value_counts())
camp = s459.campaigns; print('campaigns', camp.shape, camp.description.value_counts().to_dict())
print(camp.head(3))
ctarg = s459.campaign_targets; print('ctarg', ctarg.shape, ctarg.description.value_counts().to_dict())
cred = s459.coupon_redemptions; print('cred', cred.shape)


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, save_table, KEYS
p4 = load_saved('pred_e004.parquet'); p5 = load_saved('pred_e005.parquet')
m = p4.merge(p5, on=KEYS, suffixes=('_e4','_e5'))
print(m.shape)
m['prediction'] = 0.5*m.prediction_e4 + 0.5*m.prediction_e5
out = m[KEYS + ['prediction']]
print(out['prediction'].describe().round(1))
print(save_table(out, 'pred_e006.parquet'))
