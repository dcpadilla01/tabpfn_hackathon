
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
p = pd.Series(mdl.predict(Xev), index=ev.household_key)
e = pd.DataFrame({'y':yev.values,'p':p.values,'hh':ev.household_key.values,'eb':ev.exp4w_blend.values})
e['err']=e.p-e.y; e['abserr']=e.err.abs()
e['tb']=pd.cut(e.y,[-1,0.01,25,75,150,300,600,1e9])
print(e.groupby('tb',observed=True).agg(n=('abserr','size'),mae=('abserr','mean'),bias=('err','mean'),medp=('p','median')))
# per-household: how much of MAE is within-household volatility vs between-household?
hh_mean_y = tr.groupby('household_key').future_spend_4w.mean()
e['hhmu'] = e.hh.map(hh_mean_y)
print("MAE predict hh train-mean:", np.abs(e.hhmu-e.y).mean())
print("MAE predict exp4w_blend:", np.abs(e.eb-e.y).mean())
print("MAE predict 0.5*model+0.5*hhmu:", np.abs((0.5*e.p+0.5*e.hhmu)-e.y).mean())
print("MAE predict 0.7*model+0.3*hhmu:", np.abs((0.7*e.p+0.3*e.hhmu)-e.y).mean())
print("MAE predict 0.85*model+0.15*hhmu:", np.abs((0.85*e.p+0.15*e.hhmu)-e.y).mean())
# shrink toward household mean of PAST 4w spends (multiple lags): use feats cols lag1_spend..? use exp4w_all as hh long-run
print("MAE predict exp4w_all:", np.abs(ev.set_index('household_key')['exp4w_all'].values-e.y).mean())
# blend model with exp4w_all
ea = ev.set_index('household_key')['exp4w_all'].values
for w in [0.7,0.8,0.9]:
    print(f"blend model+exp4w_all w={w}:", round(np.abs(w*e.p+(1-w)*ea-e.y).mean(),3))
