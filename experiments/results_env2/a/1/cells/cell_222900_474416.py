
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
