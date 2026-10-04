import pandas as pd, numpy as np

def probe(v, snap):
    if snap == 95:
        print('snap', snap, '| v.day', v.day, '| v.week', v.week)
        print('households:', type(v.households), 'len', len(v.households), 'sample', list(v.households)[:3])
        print('tx shape', v.transactions.shape)
        print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)
    return pd.DataFrame(index=pd.Index(list(v.households), name='household_key'))

print('snapshot_days:', agent_api.snapshot_days())
b = agent_api.load_saved('e001_recent_spend.parquet')
print('e001 cols:', list(b.columns))
print('e001 shape:', b.shape)
print(b.head(3).T)
tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt['future_spend_4w'].describe())
_ = agent_api.build_features(probe)
print('probe done')