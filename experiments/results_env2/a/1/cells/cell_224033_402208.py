
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
