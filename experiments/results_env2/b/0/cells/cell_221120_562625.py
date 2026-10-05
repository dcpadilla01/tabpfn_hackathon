
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
