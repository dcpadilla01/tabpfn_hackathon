
import agent_api, pandas as pd
F = agent_api.load_saved('e008_candidate.parquet')
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
NEW = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84','sin_y','cos_y','day_idx']
cols = ['household_key','snapshot_day'] + E5COLS + NEW
out = F[cols]
agent_api.save_table(out, 'e008_season_basket')
print(out.shape, out.columns.tolist())
