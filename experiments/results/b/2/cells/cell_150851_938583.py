import pandas as pd, numpy as np

e5 = load_saved('e005_longrun.parquet')
e6 = load_saved('e006_fwd_profile.parquet')
cal = e5[['household_key','snapshot_day']].copy()
cal['day_idx'] = cal['snapshot_day']
cal['week_of_year'] = ((cal['snapshot_day'] + 8) // 7) % 52
cal['sin1'] = np.sin(2*np.pi*cal['week_of_year']/52); cal['cos1'] = np.cos(2*np.pi*cal['week_of_year']/52)
cal['sin2'] = np.sin(4*np.pi*cal['week_of_year']/52); cal['cos2'] = np.cos(4*np.pi*cal['week_of_year']/52)
cal['month_idx'] = cal['snapshot_day'] // 28

df = e5.merge(e6, on=['household_key','snapshot_day'], how='inner').merge(
    cal[['household_key','snapshot_day','day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx']],
    on=['household_key','snapshot_day'], how='inner')
print(df.shape)
p = save_table(df, 'e008_fwd_calendar.parquet')
print(p)