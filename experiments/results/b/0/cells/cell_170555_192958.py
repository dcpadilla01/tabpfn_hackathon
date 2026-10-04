import agent_api, pandas as pd
for n in ['e018_basestab','rich_behavioral','e006_composition','e007_lagseq','e010_decay','e011_price','e013_te_clean','e014_dm','e016_peer','season','macro','mkt_demo','rfm28']:
    try:
        df = agent_api.load_saved(n + '.parquet')
        print('==', n, df.shape)
        print(list(df.columns))
        print()
    except Exception as e:
        print('==', n, 'ERR', repr(e))
print('snapshot days:', agent_api.snapshot_days())
v = agent_api.snapshot(459)
print('txn shape @459:', v.transactions.shape)
