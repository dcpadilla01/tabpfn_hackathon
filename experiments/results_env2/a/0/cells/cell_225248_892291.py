import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category')
cat = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']
for c in cat: m[c] = m[c].cat.codes
# CV: train on snaps <= 403, eval on 431 (and 375 as second)
def cv(train_max, eval_snaps, params, blend_w=0.7, feats=None):
    feats = feats or FEATS
    tr = m[m.snapshot_day <= train_max]; te = m[m.snapshot_day.isin(eval_snaps)]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8,
             tree_method='hist', enable_categorical=True, eval_metric=None, n_jobs=4)
    p.update(params)
    mod = xgb.XGBRegressor(**p)
    mod.fit(tr[feats], tr.future_spend_4w)
    pred = np.clip(mod.predict(te[feats]), 0, None)
    if blend_w < 1:
        pred = blend_w*pred + (1-blend_w)*te.exp4w_blend.values
    res = {}
    for s in eval_snaps:
        t = te[te.snapshot_day==s]
        res[s] = (t.future_spend_4w - t.pred).abs().mean() if False else None
    out = te[['snapshot_day']].copy(); out['pred']=pred
    maes = {s: np.abs(out[out.snapshot_day==s].pred.values - te[te.snapshot_day==s].future_spend_4w.values).mean() for s in eval_snaps}
    return maes, np.mean(list(maes.values()))
base, base_m = cv(403, [375,431])
print('E007-config CV', base, round(base_m,3))
# variant: no blend (pure model)
p1, m1 = cv(403, [375,431], blend_w=1.0)
print('pure model', p1, round(m1,3))
# variant: deeper
p2, m2 = cv(403, [375,431], params=dict(max_depth=6, min_child_weight=10, n_estimators=1200))
print('deeper', p2, round(m2,3))
# variant: huber
p3, m3 = cv(403, [375,431], params=dict(objective='reg:pseudohubererror'))
print('huber', p3, round(m3,3))
