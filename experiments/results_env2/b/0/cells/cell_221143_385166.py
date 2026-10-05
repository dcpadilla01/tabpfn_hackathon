
import numpy as np, pandas as pd
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
cands = ['spend_14','spend_28','spend_56','spend_84','spend_112','spend_182','spend_364','spend_ly',
         'spend_7','dec_14','dec_28','dec_56','dec_112','ly_future28','ly_trail28','weekly_mean_12',
         'weekly_std_12','weekly_max_12','spend_cur_wk','total_spend','avg_weekly_all','z_dec_7','z_dec_224','lag1','lag2','lag3','lag4','ly_lag','lag_mean4','lag_std4']
for c in cands:
    s = pd.to_numeric(t6[c], errors='coerce')
    print(c, 'nan:', s.isna().sum(), 'min:', s.min(), 'max:', s.max())
