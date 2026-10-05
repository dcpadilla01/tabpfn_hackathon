import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet')
TOP45=['d_ewma_spend_hl28','d_ewma_spend_hl56','d_ewma_spend_hl14','d_ewma_spend_hl112','trips_84','avg28_all','total_all','d_block_std','days_84','spend_182','blk_1','days_28','spend_28','n_depts_28d','spend_rate_182','trips_56','trips_28','trips_112','d_trips_14d','d_spend_14d','d_block_max','d_zero_wk_streak_10w','d_wk_active_12w','std6','items_mean_84','days_56','d_trips_7d','trend_28_84','slope6','spend_365','spend_56','d_gap_mean_84d','days_365','classification_1','kid_category_desc','blk_5','d_gap_std_84d','days_182','trips_365','blk_11','blk_9','spend_s392','classification_3','spend_rate_84','spend_84']
NEW=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','rec_lo','rec_hi','lapsed','act_ewma28','act_share_26w','exp_spend']
keep=['household_key','snapshot_day']+TOP45+NEW
tab=d[keep].copy()
print(tab.shape, 'missing:', tab.isna().mean().mean().round(3))
agent_api.save_table(tab,'e011_lapsed.parquet')
print('saved', agent_api.snapshot_days())
