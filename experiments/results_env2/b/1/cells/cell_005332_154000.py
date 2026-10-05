import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e013_union.parquet').copy()   # best table (val MAE 60.703)
base = set(df.columns)
def col(c, fill=0.0):
    return df[c].astype(float).fillna(fill).values if c in base else np.zeros(len(df))

m_day = col('sp364').sum() / max((df['sp364'].astype(float)>0).sum()*364.0, 1.0)
new = pd.DataFrame(index=df.index)
# concave transforms of heavy-tailed levels -> closer to conditional median under MAE
for c in ['sp28','sp84','sp364','sp28_rate','sp364_rate','wksp_mean','z_med4w_hist','z_mean_week_spend_all']:
    new['sq_'+c] = np.sqrt(np.clip(col(c),0,None))
    new['lg_'+c] = np.log1p(np.clip(col(c),0,None))
# empirical-Bayes shrunk expected next-4w spend (rate shrunk to global daily rate)
for w,k in [(84,56),(168,56),(364,112)]:
    new['exp_eb%d'%w] = 28.0*(col('sp%d'%w) + k*m_day)/(w+k)
zr = np.clip(col('z_zero_rate_hist',0.5),0,1)
new['expz84']  = new['exp_eb84'].values *(1-zr)
new['expz364'] = new['exp_eb364'].values*(1-zr)
# interactions of level x cadence/recency
dsl = np.clip(col('days_since_last',999),0,728)
new['i_rec_level'] = col('sp28')*np.exp(-dsl/28.0)
new['i_trips_lvl'] = col('sp28_rate')*col('trips28')
new['i_act_lvl']   = col('sp84_rate')*col('nact84')
new['i_med_max']   = col('z_med4w_hist')*col('z_max4w_hist')
new['i_ew_mom']    = col('sp28_rate')*col('m_sp28_56')
new['i_wk_cv']     = col('wksp_mean')/(col('wksp_std')+1.0)
new['lvl_blend']   = 0.5*new['exp_eb84'].values + 0.25*new['exp_eb364'].values + 0.25*(col('z_mean_week_spend_all')*28.0)
new = new.replace([np.inf,-np.inf], np.nan)
df = pd.concat([df, new], axis=1)
print('added', new.shape[1], 'features; total cols', df.shape[1])
print('nonfinite in new block:', (~np.isfinite(new.values)).sum())
path = agent_api.save_table(df, 'e018_nonlin.parquet')
print(path)
