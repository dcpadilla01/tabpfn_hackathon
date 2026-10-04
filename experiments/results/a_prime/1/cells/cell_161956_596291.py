import agent_api as api, pandas as pd
for name in ['micro','nf_candidates','nf_transforms','nf_robust','nf_seasonal','nf_p1','nf_compact19','e015_peers_demo_l13fix']:
    df = api.load_saved(name+'.parquet')
    print('==',name, df.shape)
    print([c for c in df.columns if c not in ('household_key','snapshot_day')])
