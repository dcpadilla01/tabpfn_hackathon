
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
