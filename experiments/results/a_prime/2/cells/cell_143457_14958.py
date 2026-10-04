import pandas as pd, numpy as np
t1 = agent_api.load_saved('recency_agg.parquet')
t2 = agent_api.load_saved('dept_mix_recency.parquet')
print('E001 cols:', list(t1.columns))
print()
extra = [c for c in t2.columns if c not in t1.columns]
print('E002 extra cols (first 30):', extra[:30], '... total', len(extra))
tt = agent_api.train_targets()
print()
print(tt.future_spend_4w.describe())
print('share zero target:', (tt.future_spend_4w==0).mean())

def probe(view, d):
    base = agent_api.load_saved('dept_mix_recency.parquet')
    base = base[base.snapshot_day==d].set_index('household_key')
    return base.iloc[:, :2]
df = agent_api.build_features(probe)
print('probe ok:', df.shape)
