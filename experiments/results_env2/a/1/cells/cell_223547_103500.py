
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
