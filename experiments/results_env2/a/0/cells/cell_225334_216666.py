import pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
def cv(train_max, eval_snaps, params=None, blend_w=1.0):
    params = params or {}
    tr = m[m.snapshot_day <= train_max]; te = m[m.snapshot_day.isin(eval_snaps)]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8,
             tree_method='hist', n_jobs=4)
    p.update(params)
    mod = xgb.XGBRegressor(**p)
    mod.fit(tr[FEATS], tr.future_spend_4w)
    pred = np.clip(mod.predict(te[FEATS]), 0, None)
    if blend_w < 1: pred = blend_w*pred + (1-blend_w)*te.exp4w_blend.values
    maes = {s: round(float(np.abs(pred[te.snapshot_day==s] - te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in eval_snaps}
    return maes, round(np.mean(list(maes.values())),3)
t0=time.time()
print('base d4      ', cv(347,[375,431]))
print('d6 mcw10 n1200', cv(347,[375,431], dict(max_depth=6, min_child_weight=10, n_estimators=1200)))
print('d8 mcw10 n1200', cv(347,[375,431], dict(max_depth=8, min_child_weight=10, n_estimators=1200)))
print('d6 mcw20 n1500', cv(347,[375,431], dict(max_depth=6, min_child_weight=20, n_estimators=1500)))
print('d6 lr0.03 n2000', cv(347,[375,431], dict(max_depth=6, min_child_weight=10, learning_rate=0.03, n_estimators=2000)))
print('d6 sqerr     ', cv(347,[375,431], dict(objective='reg:squarederror', max_depth=6, min_child_weight=10, n_estimators=1200)))
print('elapsed', round(time.time()-t0,1))
