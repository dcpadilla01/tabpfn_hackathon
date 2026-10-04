names = ['blocks_v1','e005_full_plus_mix','e006_curated','e006_temporal','e007_robust','e008_dist','e009_demo','e010_rhythm','hist_v1','hist_v2','mix_v1','mkt_v1','mkt_v2','rhythm_v1','season_v1','structure_v1']
for n in names:
    try:
        df = agent_api.load_saved(n + '.parquet')
        print('==', n, df.shape)
        print(df.columns.tolist())
    except Exception as e:
        print(n, 'ERR', repr(e))
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', float((tt.future_spend_4w == 0).mean()))
v = agent_api.snapshot()
print('tx shape', v.transactions.shape)
print('n households at 459:', len(v.households))
