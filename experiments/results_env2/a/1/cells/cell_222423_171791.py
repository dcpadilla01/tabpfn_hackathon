
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
