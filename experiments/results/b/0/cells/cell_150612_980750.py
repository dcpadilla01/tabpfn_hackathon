
import pandas as pd, numpy as np
for name in ['e007_lagseq','e006_composition','rich_behavioral','mkt_demo','season','rfm28']:
    df = agent_api.load_saved(name + '.parquet')
    print('===', name, df.shape)
    print('|'.join(map(str, df.columns)))
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
v = agent_api.snapshot()
print('households type:', type(v.households))
print(v.households.head() if hasattr(v.households,'head') else v.households[:5])
print('day', v.day, 'week', v.week)
print(agent_api.snapshot_days())
