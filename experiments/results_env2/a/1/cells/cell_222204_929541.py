
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
