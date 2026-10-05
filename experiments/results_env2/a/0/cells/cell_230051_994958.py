import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f4 = agent_api.load_saved('feats_v4.parquet')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
DEMO = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']
def prep(f):
    m = tt.merge(f, on=['household_key','snapshot_day'], how='left')
    drop = ['index','household_key','snapshot_day','future_spend_4w']
    feats = [c for c in m.columns if c not in drop and c not in DEMO]
    m[feats] = m[feats].fillna(-1)
    for c in DEMO:
        m[c] = m[c].astype('category').cat.codes
    return m, feats + DEMO
m4, F4 = prep(f4)
m3, F3 = prep(f3)
params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
              learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
# quick clean CV check: v4 vs v3 (pure model)
for name, m, F in [('v3', m3, F3), ('v4', m4, F4)]:
    tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])]
    mod = xgb.XGBRegressor(**params).fit(tr[F], tr.future_spend_4w)
    pr = np.clip(mod.predict(te[F]), 0, None)
    maes = [round(float(np.abs(pr[te.snapshot_day==s]-te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in [375,431]]
    print(name, 'CV 375/431:', maes, round(np.mean(maes),3))
# final: train on all train snaps, predict validation
tr = m4[m4.snapshot_day <= 431]
mod = xgb.XGBRegressor(**params).fit(tr[F4], tr.future_spend_4w)
val = m4[m4.snapshot_day.isin([459,487,515,543])].copy()
val['prediction'] = np.clip(mod.predict(val[F4]), 0, None)
out = val[['household_key','snapshot_day','prediction']]
agent_api.save_table(out, 'pred_e008.parquet')
print('pred rows', out.shape, out.snapshot_day.value_counts().to_dict())
