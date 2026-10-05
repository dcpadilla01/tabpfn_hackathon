
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
