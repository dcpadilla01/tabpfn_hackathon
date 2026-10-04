import agent_api as api, pandas as pd, numpy as np
base = api.load_saved('e012_dorm.parquet')
bc = set(base.columns)

cand = api.load_saved('nf_candidates.parquet')
rob  = api.load_saved('nf_robust.parquet')
mic  = api.load_saved('micro.parquet')
sea  = api.load_saved('nf_seasonal.parquet')

pick_c = ['newm4','newm8','newm13','nwmean12','nf_pow90_ewm8','nf_pow90_ewm13','nspend28',
          'nf_nspend28_pow75','nspend14','nprods84','nwstd26','nwstd12','nf_ncv12','nf_ncv26',
          'nf_nmax_share12','nf_nratio7_28','nf_npertrip28','ntrips28','ndact28']
pick_r = ['r_med13','r_q75_13','r_q25_13','r_med4','r_maxmed13','r_zerow13','r_zerow4',
          'r_zerom13','pct_l1','pct_l4','pct_l13','g_mean_l4','g_med_l4','r_slope13']
pick_m = ['spend_7d','spend_14d','trips_7d','trips_14d','basket_med_84','basket_max_84',
          'basket_cv_84','basket_max_over_med_84','n_stockup_84','stockup_share_84','units_84',
          'unit_price_84','units_per_trip_84','weekend_share_84','morning_share_84','stores_84',
          'spend_cv_l6','spend_max_over_med_l6','gap_mean_l6','gap_max_l6','n_gap21_l6','dsl_stockup']
pick_s = ['nratio_ly','nspend_ly']

add = pd.concat([cand[['household_key','snapshot_day']+ [c for c in pick_c if c in cand.columns]],
                 rob [['household_key','snapshot_day']+ [c for c in pick_r if c in rob.columns]].drop(columns=['household_key','snapshot_day']),
                 mic [['household_key','snapshot_day']+ [c for c in pick_m if c in mic.columns]].drop(columns=['household_key','snapshot_day']),
                 sea [['household_key','snapshot_day']+ [c for c in pick_s if c in sea.columns]].drop(columns=['household_key','snapshot_day'])], axis=1)
add = add.loc[:, ~add.columns.duplicated()]
newcols = [c for c in add.columns if c not in bc]
add = add[['household_key','snapshot_day']+newcols]
print('new cols:', len(newcols))

out = base.merge(add, on=['household_key','snapshot_day'], how='left')
print('out', out.shape)
print('NaN frac worst:', out.isna().mean().sort_values(ascending=False).head(5).round(3).to_dict())
p = api.save_table(out, 'e016_smoothed.parquet')
print(p)
