
import pandas as pd, numpy as np
for name in ['deal_v1','hazard_v1','timing_v1','twin_v1','display_v1','lvl_v1','mkt_v1','comp_v1','ewma_block_v1','selfcal_v1','union_all_v1','combined_v1','e017_union_v1','e014_table','e014_retry']:
    try:
        df = load_saved(name+'.parquet')
        extra = [c for c in df.columns if c not in ('household_key','snapshot_day')]
        print('---', name, df.shape, 'n_extra', len(extra))
        print(extra[:40])
    except Exception as e:
        print('---', name, 'ERR', type(e).__name__, e)
