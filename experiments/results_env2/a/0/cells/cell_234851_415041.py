
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
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
ptr = mdl.predict(Xtr); pev = mdl.predict(Xev)

# calibration curve: median y per prediction bin
for X_, p_, y_, nm in [(Xtr,ptr,ytr,'train'),(Xev,pev,yev,'eval')]:
    b = pd.DataFrame({'p':p_,'y':y_})
    b['bin'] = pd.qcut(b.p, 10, duplicates='drop')
    print(nm); print(b.groupby('bin',observed=True).agg(med_p=('p','median'), med_y=('y','median'), mean_y=('y','mean'), n=('y','size')))

# per-household multiplicative calibration from train
tr2 = pd.DataFrame({'hh':tr.household_key.values,'p':ptr,'y':ytr.values})
cal = tr2.groupby('hh').agg(mu_p=('p','mean'), mu_y=('y','mean'))
cal['ratio'] = (cal.mu_y/np.maximum(cal.mu_p,1)).clip(0.5, 3.0)
e = pd.DataFrame({'hh':ev.household_key.values,'p':pev,'y':yev.values})
e = e.merge(cal, left_on='hh', right_index=True, how='left')
e['pcal'] = e.p*e.ratio.fillna(1)
print("MAE pcal (hh ratio calib):", np.abs(e.pcal-e.y).mean())
print("MAE p:", np.abs(e.p-e.y).mean())
# additive version
cal['diff'] = cal.mu_y-cal.mu_p
e2 = e.merge(cal['diff'], left_on='hh', right_index=True, how='left')
e2['pcal2'] = (e2.p+e2['diff']).clip(lower=0)
print("MAE pcal2 (hh additive):", np.abs(e2.pcal2-e2.y).mean())
# isotonic calibration on train preds
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(ptr, ytr)
print("MAE isotonic:", np.abs(iso.predict(pev)-yev).mean())
