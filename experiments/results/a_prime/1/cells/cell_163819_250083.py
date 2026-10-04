import agent_api as api
import pandas as pd, numpy as np

# Check target transform viability: what fraction of targets are 0 or tiny?
# Also check that the merged table's non-key columns are all numeric/categorical (no object dtype)
full = api.load_saved("e019_everything.parquet")
bad = [c for c in full.columns if full[c].dtype == object and c not in ('household_key','snapshot_day')]
print("object cols:", bad)

# quick sanity: correlation of each feature with target on train rows
tr = api.train_targets()
m = full.merge(tr, on=['household_key','snapshot_day'], how='inner')
print("merged with train targets:", m.shape)
num_cols = [c for c in full.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(full[c])]
corrs = {}
for c in num_cols:
    x = m[c]
    if x.notna().sum() > 1000:
        corrs[c] = np.corrcoef(x.fillna(x.median()), m['future_spend_4w'])[0,1]
corrs = pd.Series(corrs).sort_values()
print(corrs.tail(20))
print(corrs.head(10))