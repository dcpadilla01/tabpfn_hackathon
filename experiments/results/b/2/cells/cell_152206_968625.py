import pandas as pd
for name in ['e009_analog','e009_spline2p','e009_demo_mkt']:
    df = agent_api.load_saved(name+'.parquet')
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    print('==',name, df.shape, len(cols))
    print(cols[-40:])
    print()
