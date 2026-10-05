for name in ['selfcal_v1','churn_vol_v1','mkt_v1','comp_v1','ewma_block_v1','rfm_traj_v1','lvl_v1']:
    t = load_saved(name + '.parquet')
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    print(name, t.shape, 'feat:', cols[:30])
    print()
