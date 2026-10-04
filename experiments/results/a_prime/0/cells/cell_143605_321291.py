
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
def fn(view, sd):
    m = agent_api.load_saved('mkt_v2.parquet')
    sub = m[m.snapshot_day==sd].set_index('household_key')
    sub = sub.reindex(view.households)
    return sub[['m_spend28']]
df = agent_api.build_features(fn)
print(df.shape, df.head(3))
print('missing:', df.m_spend28.isna().mean())
