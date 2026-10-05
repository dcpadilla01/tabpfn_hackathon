
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
t7 = agent_api.load_saved('e007_ar_lags.parquet')
print('e006', t6.shape); print(list(t6.columns))
print('e007', t7.shape); print([c for c in t7.columns if c not in t6.columns])
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share train:', (tt.future_spend_4w==0).mean())


# ---- cell ----

import numpy as np, pandas as pd
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
exclude_exact = {'week','week_sin','week_cos','week_q','snapshot_day','household_key'}
exclude_suffix = ('_share','_code','_ts')  # keep ratios/shares/codes as-is
exclude_names = {'spend_28_56','spend_84_28','r_28_56','r_28_84','r_84_364','lag_ratio','ly_ratio',
                 'active_week_share','unit_price_84','trend','trend_slope_12','z_recency_norm',
                 'z_notrip28_x_rate','z_recency_x_rate','z_dec7_x_act4','z_rate84','z_last_basket'}
t = t6.copy()
logged = []
for c in t.columns:
    if c in exclude_exact or c.endswith(exclude_suffix) or c in exclude_names: continue
    s = t[c]
    if s.min() >= 0 and s.max() > 60:
        t[c] = np.log1p(s); logged.append(c)
print(len(logged), 'logged:', logged)
print(t[logged[:6]].describe().loc[['min','max']])
path = agent_api.save_table(t, 'e008_log_transform')
print(path, t.shape)


# ---- cell ----

import numpy as np, pandas as pd
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
exclude_exact = {'week','week_sin','week_cos','week_q','snapshot_day','household_key'}
exclude_suffix = ('_share','_code','_ts')
exclude_names = {'spend_28_56','spend_84_28','r_28_56','r_28_84','r_84_364','lag_ratio','ly_ratio',
                 'active_week_share','unit_price_84','trend','trend_slope_12','z_recency_norm',
                 'z_notrip28_x_rate','z_recency_x_rate','z_dec7_x_act4','z_rate84','z_last_basket'}
t = t6.copy()
logged = []
for c in t.columns:
    if c in exclude_exact or c.endswith(exclude_suffix) or c in exclude_names: continue
    s = pd.to_numeric(t[c], errors='coerce')
    if s.notna().all() and s.min() >= 0 and s.max() > 60:
        t[c] = np.log1p(s); logged.append(c)
print(len(logged), 'logged:', logged)
path = agent_api.save_table(t, 'e008_log_transform')
print(path, t.shape)


# ---- cell ----

import numpy as np, pandas as pd
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
cands = ['spend_14','spend_28','spend_56','spend_84','spend_112','spend_182','spend_364','spend_ly',
         'spend_7','dec_14','dec_28','dec_56','dec_112','ly_future28','ly_trail28','weekly_mean_12',
         'weekly_std_12','weekly_max_12','spend_cur_wk','total_spend','avg_weekly_all','z_dec_7','z_dec_224','lag1','lag2','lag3','lag4','ly_lag','lag_mean4','lag_std4']
for c in cands:
    s = pd.to_numeric(t6[c], errors='coerce')
    print(c, 'nan:', s.isna().sum(), 'min:', s.min(), 'max:', s.max())
