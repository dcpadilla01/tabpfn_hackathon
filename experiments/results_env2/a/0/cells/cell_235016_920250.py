
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)

# in-sample MAE to gauge underfit
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
print("in-sample MAE:", np.abs(mdl.predict(Xtr)-ytr).mean(), "| eval MAE:", np.abs(mdl.predict(Xev)-yev).mean())

# 1. seed-bagged median ensemble
preds=[]
for s in [1,2,3]:
    p_ = dict(base); p_['subsample']=0.7+0.1*s; 
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, random_state=s, **p_).fit(Xtr,ytr)
    preds.append(pd.Series(mm.predict(Xev), index=ev.household_key))
P = pd.concat(preds, axis=1)
print("bag mean MAE:", np.abs(P.mean(1).values-yev).mean(), "| bag median MAE:", np.abs(P.median(1).values-yev).mean())

# 2. residual on blend
eb_tr = tr.exp4w_blend.values; eb_ev = ev.exp4w_blend.values
mr = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr-eb_tr)
pr = mr.predict(Xev)+eb_ev
print("residual-on-blend MAE:", np.abs(pr-yev).mean())

# 3. lower lr, more trees
for ne,lr,dp in [(800,0.04,5),(600,0.05,4),(500,0.06,6)]:
    p_=dict(base); p_['learning_rate']=lr; p_['max_depth']=dp
    mm = xgb.XGBRegressor(n_estimators=ne, tree_method='hist', n_jobs=4, verbosity=0, **p_).fit(Xtr,ytr)
    print(f"ne={ne} lr={lr} depth={dp}: MAE", round(np.abs(mm.predict(Xev)-yev).mean(),3))
