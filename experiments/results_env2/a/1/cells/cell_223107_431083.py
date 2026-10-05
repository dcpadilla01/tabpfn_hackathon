
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
