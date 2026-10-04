import agent_api, pandas as pd, numpy as np
for n in ['rich_behavioral','e006_composition','e007_lagseq','e010_decay','e011_price','e013_te_clean','e014_dm','e016_peer']:
    df = agent_api.load_saved(n + '.parquet')
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    print('==', n, len(cols))
    print(cols)
    print()