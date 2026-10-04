import agent_api as api
import pandas as pd, numpy as np

key = ['household_key','snapshot_day']
tt = api.train_targets()
e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=key, how='inner').copy()
y = m['future_spend_4w'].values

# residualize target on E015's strongest predictors
X = m[['e13','ew28','usual13_4','mfly','spend_84','b75']].copy().fillna(0)
Xm = np.c_[np.ones(len(X)), X.values]
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
resid = y - Xm @ beta

# candidate pool of NEW features (not in E015)
e10 = api.load_saved('e010_composite.parquet').set_index(key)
e01 = api.load_saved('e001_history.parquet').set_index(key)
e04 = api.load_saved('e004_long_hist.parquet').set_index(key)
e03 = api.load_saved('e003_dept_mix.parquet').set_index(key)
bf  = api.baseline_features().set_index(key)

pool = {
 'act_persist': e10['act_persist'], 'us_stab': e10['us_stab'], 'log_act_usual': e10['log_act_usual'],
 'spend_7': e01['spend_7'], 'spend_28': e01['spend_28'], 'log_spend_28': e01['log_spend_28'],
 'days_since_last': e01['days_since_last'], 'qty_28': e01['qty_28'], 'spend_prev28': e01['spend_prev28'],
 'trend28': e01['trend28'], 'spend_all': e01['spend_all'], 'spend_ly28': e01['spend_ly28'],
 'nb_56': e04['nb_56'], 'std_wk8': e04['std_wk8'], 'max_wk8': e04['max_wk8'], 'wk_rate_all': e04['wk_rate_all'],
 'd84_SALAD BAR': e03['d84_SALAD BAR'], 'd84_GARDEN CENTER': e03['d84_GARDEN CENTER'],
 'd84_DELI': e03['d84_DELI'], 'd84_HBC': e03['d84_HBC'], 'd84_MEAT-PCKGD': e03['d84_MEAT-PCKGD'],
 'dept_entropy_84': e03['dept_entropy_84'], 'zero_week_share_84': e03['zero_week_share_84'],
 'spend_vol_84': e03['spend_vol_84'],
 'classification_2': bf['classification_2'], 'snapshot_day_index': bf['snapshot_day_index'],
 'week_of_year': bf['week_of_year'],
}
pool = pd.DataFrame(pool)

# residual correlations on train rows
idx = m.set_index(key).index
p_tr = pool.loc[idx]
cors = {}
for c in pool.columns:
    x = pd.to_numeric(p_tr[c], errors='coerce')
    if x.notna().sum() < 500:
        cors[c] = np.nan
    else:
        cors[c] = np.corrcoef(x.fillna(x.median()).values, resid)[0,1]
cs = pd.Series(cors)
print('residual corrs:'); print(cs.sort_values(key=lambda s: s.abs(), ascending=False).round(4))

# select: |resid corr| >= 0.018, plus always calendar + classification_2 + spend_28
sel = [c for c in cs.index if (abs(cs[c]) >= 0.018)]
for c in ['classification_2','snapshot_day_index','week_of_year','spend_28']:
    if c not in sel: sel.append(c)
print('\nselected:', sel, len(sel))

# build final table: E015 base (drop constant 'streak') + selected
base = e15.drop(columns=['streak'])
add = pool[sel].reset_index()
final = base.merge(add, on=key, how='left')
print('final shape:', final.shape, ' dup rows:', final.duplicated(key).sum())
print('na frac of new cols (max):', final[sel].isna().mean().max().round(3))
path = api.save_table(final, 'e017_gapfill')
print('saved:', path)