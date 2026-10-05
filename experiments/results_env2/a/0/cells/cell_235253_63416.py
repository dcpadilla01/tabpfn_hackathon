
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day']
for c in m.columns:
    if str(m[c].dtype)=='category': m[c] = m[c].cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.55)
def q(**kw):
    p=dict(base); p.update(kw)
    return xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **p)

# with interactions
I_tr = np.column_stack([tr.spend_28*tr.tenure, tr.spend_28*tr.trips_28, tr.exp4w_blend*tr.tenure]).astype(float)
I_ev = np.column_stack([ev.spend_28*ev.tenure, ev.spend_28*ev.trips_28, ev.exp4w_blend*ev.tenure]).astype(float)
Xtr2, Xev2 = np.column_stack([Xtr,I_tr]), np.column_stack([Xev,I_ev])
x1 = q().fit(Xtr2, ytr); p1 = x1.predict(Xev2)
hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, min_samples_leaf=60, loss='absolute_error', random_state=0).fit(Xtr2, ytr)
p2 = hg.predict(Xev2)
print("combo alpha55+int, xgb:", round(np.abs(p1-yev).mean(),4))
print("combo blend 0.7/0.3:", round(np.abs(0.7*p1+0.3*p2-yev).mean(),4))
print("combo blend 0.6/0.4:", round(np.abs(0.6*p1+0.4*p2-yev).mean(),4))
print("combo blend 0.5/0.5:", round(np.abs(0.5*p1+0.5*p2-yev).mean(),4))
