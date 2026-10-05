
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
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
def q(**kw):
    p=dict(base); p.update(kw)
    return xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **p)
mdl = q().fit(Xtr, ytr); pev = mdl.predict(Xev)
print("refit WITH snap feats MAE:", round(np.abs(pev-yev).mean(),4))

for a in [0.45,0.55]:
    mm=q(quantile_alpha=a).fit(Xtr,ytr); print(f"alpha={a}:", round(np.abs(mm.predict(Xev)-yev).mean(),4))

# multiplicative residual on exp4w_blend
eb_tr, eb_ev = tr.exp4w_blend.values, ev.exp4w_blend.values
mm = q().fit(Xtr, np.log((ytr+1)/(eb_tr+1)))
pm = np.maximum(eb_ev+1,0)*np.exp(mm.predict(Xev))-1
print("mult-resid-on-blend:", round(np.abs(pm-yev).mean(),4))

# ratio target: y / snapshot global mean
gm = tr.groupby('snapshot_day').future_spend_4w.mean()
mm = q().fit(Xtr, ytr/tr.snapshot_day.map(gm).values)
pm2 = mm.predict(Xev)*ev.snapshot_day.map(gm).values
print("ratio-target:", round(np.abs(pm2-yev).mean(),4))

# HistGB blend
hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=None, min_samples_leaf=60, loss='absolute_error', random_state=0).fit(Xtr, ytr)
ph = hg.predict(Xev)
print("HistGB MAE:", round(np.abs(ph-yev).mean(),4))
for w in [0.3,0.5]:
    print(f"blend xgb+hist w={w}:", round(np.abs(w*pev+(1-w)*ph-yev).mean(),4))
