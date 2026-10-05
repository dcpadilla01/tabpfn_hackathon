
import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 200)

f = agent_api.load_saved('e004_features.parquet')
print('e004_features', f.shape)
print(f.dtypes.to_string())
print(f.head(3).to_string())

p = agent_api.load_saved('e004_preds.parquet')
print('\ne004_preds', p.shape, p.columns.tolist())
print(p.prediction.describe())
print('neg preds:', (p.prediction < 0).sum())

tt = agent_api.train_targets()
print('\ntrain_targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share:', (tt.future_spend_4w == 0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['size','mean','median']).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
print('xgb', xgb.__version__)
F = agent_api.load_saved('e004_features.parquet')
T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
print('merged', D.shape, 'missing target', D.future_spend_4w.isna().sum())
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
print('n feats', len(feats))

# internal validation: train snaps < V, val snap V
def run(V, params, target='y', blend=None, wdecay=277, nround=1200, verbose=False):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / wdecay)
    if target == 'log1p':
        ytr = np.log1p(tr.future_spend_4w)
    else:
        ytr = tr.future_spend_4w
    m = xgb.XGBRegressor(**params)
    m.fit(tr[feats], ytr, sample_weight=w, verbose=False)
    pv = m.predict(va[feats])
    if target == 'log1p': pv = np.expm1(pv)
    pv = np.clip(pv, 0, None)
    if blend is not None:
        pv = blend*pv + (1-blend)*va['spend_28'].values
    mae = np.abs(pv - va.future_spend_4w.values).mean()
    return mae, m, pv

base = dict(n_estimators=1200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

t0=time.time()
for name, kw in [
    ('quant d4', {}),
    ('quant d3', dict(max_depth=3)),
    ('L1 d4', dict(objective='reg:absoluteerror')),
    ('sq log1p d4', dict(objective='reg:squarederror'), ),
]:
    p = dict(base); p.update(kw)
    tgt = 'log1p' if 'log1p' in name else 'y'
    mae,_,_ = run(431, p, target=tgt)
    print(f'{name:14s} MAE@431 {mae:.3f}  ({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=1200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def run(V, params, blend=0.0, wdecay=277, nround=None, lr=None):
    p = dict(params)
    if nround: p['n_estimators']=nround
    if lr: p['learning_rate']=lr
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / wdecay)
    m = xgb.XGBRegressor(**p); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
    pv = np.clip(m.predict(va[feats]), 0, None)
    if blend>0: pv = blend*pv + (1-blend)*va['spend_28'].values
    return np.abs(pv - va.future_spend_4w.values).mean()

t0=time.time()
# blends on 431 and 403
for V in (431, 403):
    for b in (0.0, 0.15, 0.3, 0.5):
        print(f'V={V} blend={b}: {run(V, base, blend=b):.3f}')
print(f'({time.time()-t0:.0f}s)')
# lr / rounds
for nround, lr in [(600,0.08),(1200,0.05),(2400,0.03),(3600,0.02)]:
    print(f'V=431 n={nround} lr={lr}: {run(431, base, nround=nround, lr=lr):.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

V=431
tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
w = 0.5 ** ((V - tr.snapshot_day) / 277)
t0=time.time()
m = xgb.XGBRegressor(**base); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
pv = np.clip(m.predict(va[feats]), 0, None)
print('quant:', np.abs(pv - va.future_spend_4w.values).mean())

imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
print(imp.head(25).to_string())
print('bottom:', imp.tail(8).index.tolist())

# error by bucket
err = pd.DataFrame({'y': va.future_spend_4w.values, 'p': pv})
err['b'] = pd.cut(err.y, [-1,0,30,80,200,1e9])
print(err.groupby('b', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.p-g.y).mean(),'bias':(g.p-g.y).mean()}), include_groups=False))

# two-stage: P(y>0) classifier * prediction
clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='binary:logistic')
clf.fit(tr[feats], (tr.future_spend_4w>0).astype(int), sample_weight=w, verbose=False)
pz = clf.predict_proba(va[feats])[:,1]
for thr in (0.0, 0.1, 0.2):
    pv2 = pv * (pz>thr)
    print(f'two-stage thr={thr}: {np.abs(pv2 - va.future_spend_4w.values).mean():.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def fit_pred(V, params, target='raw'):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / 277)
    y = tr.future_spend_4w if target=='raw' else np.log1p(tr.future_spend_4w)
    m = xgb.XGBRegressor(**params); m.fit(tr[feats], y, sample_weight=w, verbose=False)
    p = m.predict(va[feats])
    if target!='raw': p = np.expm1(p)
    return va, np.clip(p, 0, None)

t0=time.time()
res = {}
for V in (431, 403):
    # base alpha 0.5
    va, pv = fit_pred(V, base)
    res[(V,'a50')] = (va, pv)
    print(f'V={V} alpha .50: {np.abs(pv-va.future_spend_4w.values).mean():.3f}')
    # post-hoc upper-tail boost
    for thr, mult in [(150,1.15),(150,1.3),(100,1.2)]:
        p2 = pv*(1+(mult-1)*(pv>thr))
        print(f'   boost>{thr}x{mult}: {np.abs(p2-va.future_spend_4w.values).mean():.3f}')
    # zero-gate rule: spend_84==0
    z = va['spend_84'].values==0
    p3 = pv.copy(); p3[z] = 0
    print(f'   gate spend84==0 ->0 ({z.sum()} rows): {np.abs(p3-va.future_spend_4w.values).mean():.3f}')
    # alpha variants
    for a in (0.55, 0.60):
        p = dict(base); p['quantile_alpha']=a
        va2, pv2 = fit_pred(V, p)
        res[(V,f'a{int(a*100)}')] = (va2, pv2)
        print(f'V={V} alpha {a}: {np.abs(pv2-va2.future_spend_4w.values).mean():.3f}')
    # log-space median
    va3, pv3 = fit_pred(V, base, target='log')
    res[(V,'log')] = (va3, pv3)
    print(f'V={V} log-median: {np.abs(pv3-va3.future_spend_4w.values).mean():.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, reg_lambda=1.0, tree_method='hist', n_jobs=8,
            objective='reg:quantileerror', quantile_alpha=0.5)

def preds(V, params):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / 277)
    m = xgb.XGBRegressor(**params); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
    return np.clip(m.predict(va[feats]), 0, None), va

t0=time.time()
for V in (431, 403):
    P = []
    for cfg in [dict(max_depth=4), dict(max_depth=3), dict(max_depth=5, min_child_weight=40),
                dict(max_depth=4, subsample=0.7, colsample_bytree=0.7)]:
        for seed in (1, 2):
            p = dict(base); p.update(cfg); p['random_state']=seed
            pv, va = preds(V, p); P.append(pv)
    ens = np.mean(P, axis=0)
    mae = np.abs(ens - va.future_spend_4w.values).mean()
    # median of members
    med = np.median(P, axis=0)
    print(f'V={V} ens8(mean): {mae:.3f} | ens8(median): {np.abs(med-va.future_spend_4w.values).mean():.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, time

def fn(view, snap):
    hh = view.households
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(hh))][['household_key','basket_id','day','sales_value','product_id']]
    prod = view.table('products')[['product_id','department']]
    tx = tx.merge(prod, on='product_id', how='left')
    g = tx.groupby(['household_key','basket_id'], as_index=False).agg(day=('day','first'), spend=('sales_value','sum'))
    out = pd.DataFrame(index=hh)
    # disjoint 28d windows s1..s4
    for i,(lo,hi) in enumerate([(snap-27,snap),(snap-55,snap-28),(snap-83,snap-56),(snap-111,snap-84)]):
        m = g[(g.day>=lo)&(g.day<=hi)].groupby('household_key').spend.sum()
        out[f's{i+1}'] = m.reindex(hh).fillna(0.0)
    s = out[['s1','s2','s3','s4']].values
    out['seq_mean'] = s.mean(1); out['seq_std'] = s.std(1)
    out['seq_cv'] = out['seq_std']/(out['seq_mean']+1)
    out['ratio_s1_s3'] = out.s1/(out.s3+1)
    out['ratio_s1_s4'] = out.s1/(out.s4+1)
    # dept-level: spend last 28d, days since last dept purchase
    top = ['GROCER','PRODUC','MEAT','DRUG G','DELI']
    t28 = tx[(tx.day>snap-28)&(tx.department.isin(top))]
    out['dept28_tot'] = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    for d in top:
        td = tx[tx.department==d]
        out[f'dsp28_{d[:4]}'] = td[td.day>snap-28].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        last = td.groupby('household_key').day.max()
        out[f'drec_{d[:4]}'] = (snap-last).reindex(hh).fillna(999.0)
    # basket gaps in last 112d
    b = g[(g.day>snap-112)].sort_values(['household_key','day'])
    b['gap'] = b.groupby('household_key').day.diff()
    gg = b.groupby('household_key').gap
    out['gap_mean'] = gg.mean().reindex(hh)
    out['gap_std'] = gg.std().reindex(hh)
    out['gap_max'] = gg.max().reindex(hh)
    out['gap_nbig'] = gg.apply(lambda x:(x>21).sum()).reindex(hh).fillna(0.0)
    lastb = b.groupby('household_key').day.max()
    out['last_gap'] = (snap-lastb).reindex(hh)
    # weekly activity streaks, last 12 weeks
    wk = (snap+8)//7
    txb = g.copy(); txb['w'] = (txb.day+8)//7
    aw = txb[(txb.w>wk-12)&(txb.w<=wk)].groupby(['household_key','w']).size().unstack(fill_value=0)
    aw = aw.reindex(hh).fillna(0.0)
    A = (aw>0).values.astype(int)
    cur_inact = np.zeros(len(hh)); streak=0; best=0
    for j in range(A.shape[1]):
        col = A[:,j]
        cur_inact = np.where(col==0, cur_inact+1, 0)
        streak = np.where(col==1, streak+1, 0)
        best = np.maximum(best, streak)
    out['wk_inact_streak'] = cur_inact; out['wk_best_streak'] = best
    out['spend28_vs_84'] = out.s1/(out.s1*0+1)  # placeholder replaced below
    return out.drop(columns=['spend28_vs_84'])

t0=time.time()
NF = agent_api.build_features(fn)
print('new feats', NF.shape, f'{time.time()-t0:.0f}s')
print(NF.head(3).to_string())
agent_api.save_table(NF, 'e005_newfeats.parquet')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
NF = agent_api.load_saved('e005_newfeats.parquet')
D = F.merge(T, on=['household_key','snapshot_day'], how='left').merge(
    NF.drop(columns=[]), on=['household_key','snapshot_day'], how='left')
newf = [c for c in NF.columns if c not in ('household_key','snapshot_day')]
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
print('nan in new feats:', D[newf].isna().sum().sum())
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def run(V, feats, params=None):
    p = dict(base); p.update(params or {})
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / 277)
    m = xgb.XGBRegressor(**p); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
    pv = np.clip(m.predict(va[feats]), 0, None)
    return np.abs(pv - va.future_spend_4w.values).mean(), m

t0=time.time()
for V in (431, 403):
    m0,_ = run(V, oldf); m1,_ = run(V, oldf+newf)
    print(f'V={V}: E004 feats {m0:.3f} | +new {m1:.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
NF = agent_api.load_saved('e005_newfeats.parquet')
NF = NF.fillna({'gap_mean':0,'gap_std':0,'gap_max':0,'last_gap':0})
NF['gap_mean']=NF.gap_mean.fillna(0); NF['gap_std']=NF.gap_std.fillna(0); NF['gap_max']=NF.gap_max.fillna(0); NF['last_gap']=NF.last_gap.fillna(0)
D = F.merge(T, on=['household_key','snapshot_day'], how='left').merge(NF, on=['household_key','snapshot_day'], how='left')
newf = [c for c in NF.columns if c not in ('household_key','snapshot_day')]
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
print('nan:', D[newf].isna().sum().sum())
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def run(V, feats):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / 277)
    m = xgb.XGBRegressor(**base); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
    pv = np.clip(m.predict(va[feats]), 0, None)
    return np.abs(pv - va.future_spend_4w.values).mean(), m

t0=time.time()
subsets = {
 'all_new': newf,
 'seq+gap': [c for c in newf if c.startswith(('s1','s2','s3','s4','seq','ratio','gap','last_gap'))],
 'dept': [c for c in newf if c.startswith(('dsp','drec','dept28'))],
 'seq+dept': [c for c in newf if c.startswith(('s1','s2','s3','s4','seq','ratio','dsp','drec','dept28'))],
 'wk': [c for c in newf if c.startswith('wk_')],
}
for V in (431, 403):
    m0,_ = run(V, oldf)
    line = f'V={V}: base {m0:.3f} |'
    for nm, sub in subsets.items():
        m1,_ = run(V, oldf+sub)
        line += f' {nm} {m1:.3f} |'
    print(line)
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def run(V, feats, decay=277, mode='raw'):
    tr = D[D.snapshot_day < V].copy(); va = D[D.snapshot_day == V].copy()
    w = 0.5 ** ((V - tr.snapshot_day) / decay)
    p = dict(base)
    if 'snap' in mode:
        tr['snap'] = tr.snapshot_day.astype(float); va['snap'] = va.snapshot_day.astype(float)
        feats = feats + ['snap']
    if mode=='ratio':
        b_tr = tr['spend_seas_364'].values; b_va = va['spend_seas_364'].values
        ytr = tr.future_spend_4w.values/(b_tr+50); yva = va.future_spend_4w.values
        m = xgb.XGBRegressor(**p); m.fit(tr[feats], ytr, sample_weight=w, verbose=False)
        pv = np.clip(m.predict(va[feats]),0,None)*(b_va+50)
    elif mode=='delta':
        b_tr = tr['spend_seas_364'].values; b_va = va['spend_seas_364'].values
        ytr = tr.future_spend_4w.values - b_tr
        m = xgb.XGBRegressor(**p); m.fit(tr[feats], ytr, sample_weight=w, verbose=False)
        pv = np.clip(m.predict(va[feats]),0,None) + b_va
    else:
        m = xgb.XGBRegressor(**p); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
        pv = np.clip(m.predict(va[feats]),0,None)
    return np.abs(pv - va.future_spend_4w.values).mean()

t0=time.time()
for V in (431, 403):
    print(f'V={V}: raw {run(V,oldf):.3f} | raw+snapfeat {run(V,oldf,mode="raw snap"):.3f} | decay140 {run(V,oldf,decay=140):.3f} | decay90 {run(V,oldf,decay=90):.3f} | ratio {run(V,oldf,mode="ratio"):.3f} | delta {run(V,oldf,mode="delta"):.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def preds(V, params, decay):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / decay)
    m = xgb.XGBRegressor(**params); m.fit(tr[feats:=oldf], tr.future_spend_4w, sample_weight=w, verbose=False)
    return np.clip(m.predict(va[oldf]), 0, None), va

t0=time.time()
for V in (431, 403):
    P=[]
    p = dict(base); p['objective']='reg:pseudohubererror'; p['huber_slope']=5
    ph,_ = preds(V, p, 140)
    print(f'V={V} huber5 d140: {np.abs(ph-va_y if False else 0):.3f}' if False else '', end='')
    va = D[D.snapshot_day==V]; y = va.future_spend_4w.values
    print(f'V={V} huber5 d140: {np.abs(ph-y).mean():.3f}')
    for dec in (140, 200):
        pv,_ = preds(V, base, dec)
        print(f'V={V} quant d{dec}: {np.abs(pv-y).mean():.3f}')
        P.append(pv)
    ens = np.mean(P, axis=0)
    print(f'V={V} ens(140,200): {np.abs(ens-y).mean():.3f}')
print(f'({time.time()-t0:.0f}s)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet')
T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
tr = D[D.future_spend_4w.notna()].copy()
va = F[F.snapshot_day >= 459].copy()
feats = [c for c in F.columns if c not in ('household_key','snapshot_day')]
print('train rows', len(tr), 'val rows', len(va), 'val days', sorted(va.snapshot_day.unique()))
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)
t0=time.time()
P = np.zeros(len(va))
for decay in (140, 200):
    w = 0.5 ** ((459 - tr.snapshot_day) / decay)  # weight by distance to first val day
    for seed in (1, 2):
        p = dict(base); p['random_state'] = seed
        m = xgb.XGBRegressor(**p)
        m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
        P += np.clip(m.predict(va[feats]), 0, None)
        print(f'decay={decay} seed={seed} done ({time.time()-t0:.0f}s)')
P /= 4
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = P
print(out.prediction.describe())
agent_api.save_table(out, 'e005_preds.parquet')
