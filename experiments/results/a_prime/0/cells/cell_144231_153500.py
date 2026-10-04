import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
mkt = A.load_saved('mkt_v2.parquet')
merged = A.load_saved('e006_temporal.parquet')
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
sub = merged[newf].apply(pd.to_numeric, errors='coerce')
print('infs per col (top):')
inf_counts = np.isinf(sub.values).sum(0)
print({c: int(n) for c, n in zip(newf, inf_counts) if n > 0})
print('\nmax abs per col:')
mx = sub.abs().max()
print(mx.sort_values(ascending=False).head(15))
print('\n99.9% quantile abs:')
q = sub.abs().quantile(0.999)
print(q.sort_values(ascending=False).head(15))