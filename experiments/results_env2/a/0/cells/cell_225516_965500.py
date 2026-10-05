import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])].copy()
p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
         learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
mod = xgb.XGBRegressor(**p); mod.fit(tr[FEATS], tr.future_spend_4w)
te['pred'] = np.clip(mod.predict(te[FEATS]),0,None)
te['y'] = te.future_spend_4w
print('=== by days_since_last bucket ===')
te['dsl_b'] = pd.cut(te.days_since_last, [-1,7,14,21,28,42,60,10000])
g = te.groupby('dsl_b', observed=True).apply(lambda d: pd.Series({'n':len(d),'y_med':d.y.median(),'pred_med':d.pred.median(),'mae':np.abs(d.y-d.pred).mean(),'zero_frac':(d.y==0).mean()}))
print(g.round(2))
print('=== by exp4w_blend decile ===')
te['eb'] = pd.qcut(te.exp4w_blend, 8, duplicates='drop')
g2 = te.groupby('eb', observed=True).apply(lambda d: pd.Series({'n':len(d),'y_med':d.y.median(),'pred_med':d.pred.median(),'mae':np.abs(d.y-d.pred).mean()}))
print(g2.round(2))
print('=== zero-target rows ===')
z = te[te.y==0]
print('n', len(z), 'pred mean', z.pred.mean().round(2), 'pred med', z.pred.median().round(2), 'sum MAE contrib', np.abs(z.y-z.pred).mean().round(2))
nz = te[te.y>0]
print('nonzero n', len(nz), 'MAE', np.abs(nz.y-nz.pred).mean().round(2))
print('=== if pred=0 for dsl>28 ===')
alt = te.pred.copy(); alt[te.days_since_last>28]=0
print('mae', np.abs(alt-te.y).mean().round(3), 'vs', np.abs(te.pred-te.y).mean().round(3))
print('=== if pred=0 for dsl>42 ===')
alt2 = te.pred.copy(); alt2[te.days_since_last>42]=0
print('mae', np.abs(alt2-te.y).mean().round(3))
print('dsl>28 zero frac:', (te[te.days_since_last>28].y==0).mean().round(3), 'n=', (te.days_since_last>28).sum())
