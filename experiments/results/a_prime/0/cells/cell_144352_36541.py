import agent_api as A
import pandas as pd
merged = A.load_saved('e006_temporal.parquet')
mkt = A.load_saved('mkt_v2.parquet')
curated = ['slope','p12','yoy_diff','seas_ratio','nb12','p5','w5','w2','p6','w_slope','b_mean','nb3','p11','nb11','p3','w8','nb5','w7','nb4','p1','n_blocks_full','w3','nb_std','nb_max','w_cv','nb2','p2','nb1','w1','p1_div_bmean']
keep = ['household_key','snapshot_day'] + [c for c in mkt.columns if c not in ('household_key','snapshot_day')] + curated
out = merged[keep]
print(out.shape)
p = A.save_table(out, 'e006_curated')
print(p)