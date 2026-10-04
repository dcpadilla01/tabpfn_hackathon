import agent_api as api

# What's in the current best table and the prepared-but-unused tables?
for name in ["e016_smoothed", "e017_v2", "e017_disc_seasonal", "e018_timing_hazard"]:
    df = api.load_saved(name + ".parquet")
    print("==", name, df.shape)
    print(list(df.columns))
    print()

# ---- cell ----
import agent_api as api

base = api.load_saved("e018_timing_hazard.parquet")   # E016 best table (157 feats)
demo = api.load_saved("e011_demo.parquet")
v2   = api.load_saved("e017_v2.parquet")

print("base cols with demo-ish names:", [c for c in base.columns if 'class' in c or 'home' in c or 'kid' in c or 'demo' in c])
print("demo table cols:", list(demo.columns))
print("v2 extras:", [c for c in v2.columns if c not in base.columns])
print("base dup rows on keys:", base.duplicated(['household_key','snapshot_day']).sum())
print("demo dup rows on keys:", demo.duplicated(['household_key','snapshot_day']).sum())
print("demo dtypes:", demo.dtypes.to_dict())

# ---- cell ----
import agent_api as api

for name in ["e014_demo_l13fix", "e015_peers_demo_l13fix", "e012_dorm", "e013_peers"]:
    df = api.load_saved(name + ".parquet")
    demo_cols = [c for c in df.columns if 'class' in c or 'home' in c or 'kid' in c or 'has_demo' in c]
    print(name, df.shape, "| demo cols:", demo_cols)

# check overlap of e016 base with e014 table
e14 = api.load_saved("e014_demo_l13fix.parquet")
e16 = api.load_saved("e016_smoothed.parquet")
print("\ne14 extras vs e16:", [c for c in e14.columns if c not in e16.columns])
print("e16 extras vs e14:", [c for c in e16.columns if c not in e14.columns])

# ---- cell ----
import agent_api as api
import pandas as pd

e16 = api.load_saved("e016_smoothed.parquet")   # 127 cols incl keys
e14 = api.load_saved("e014_demo_l13fix.parquet")  # has demo cols
demo_cols = ['classification_1','classification_2','classification_3','classification_4',
             'classification_5','homeowner_desc','kid_category_desc','has_demographics']
extra14 = ['has_real_l13','l13_over_recent']

merged = e16.merge(e14[['household_key','snapshot_day']+demo_cols+extra14],
                   on=['household_key','snapshot_day'], how='left')
print(merged.shape)
print("dup:", merged.duplicated(['household_key','snapshot_day']).sum())
print("demo coverage:", merged['has_demographics'].mean())
print(merged['classification_1'].value_counts(dropna=False).head())
path = api.save_table(merged, "e019_full_merged")
print(path)

# ---- cell ----
import agent_api as api

# peek at what the model sees: is the model gradient boosting? unknown; just check target distribution
tr = api.train_targets()
print(tr.shape, tr['future_spend_4w'].describe())
print("val snapshots:", api.snapshot_days())

# also check: how many households per snapshot, and spend scale
import numpy as np
print(tr.groupby('snapshot_day')['future_spend_4w'].mean())

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

# Check target transform viability: what fraction of targets are 0 or tiny?
tr = api.train_targets()
y = tr['future_spend_4w']
print("zeros:", (y==0).mean(), " <5:", (y<5).mean(), " <20:", (y<20).mean())
print(np.log1p(y).describe())

# Also check skew of spend_l1 (main predictor)
e16 = api.load_saved("e016_smoothed.parquet")
print(e16['spend_l1'].describe())
print("corr log1p(y) with log1p(spend_l1):", np.corrcoef(np.log1p(tr['future_spend_4w']), np.log1p(e16.merge(tr[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='right')['spend_l1'].fillna(0)))[0,1])

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

# Check overlap of e017_v2 with e018_timing_hazard (E016's table)
v2 = api.load_saved("e017_v2.parquet")
e18 = api.load_saved("e018_timing_hazard.parquet")
print("v2 extras vs e18:", [c for c in v2.columns if c not in e18.columns])
print("e18 extras vs v2:", [c for c in e18.columns if c not in v2.columns])

# Check e013 peers extras
e13 = api.load_saved("e013_peers.parquet")
print("e13 extras vs e18:", [c for c in e13.columns if c not in e18.columns])

# ---- cell ----
import agent_api as api
import pandas as pd

e16  = api.load_saved("e016_smoothed.parquet")
e14  = api.load_saved("e014_demo_l13fix.parquet")
e13  = api.load_saved("e013_peers.parquet")
v2   = api.load_saved("e017_v2.parquet")

demo_cols = ['classification_1','classification_2','classification_3','classification_4',
             'classification_5','homeowner_desc','kid_category_desc','has_demographics']
l13_cols = ['has_real_l13','l13_over_recent']
peer_cols = ['peer_recent28','peer_recent28_med','peer_ratio','peer_ratio2','peer_p90','peer_p10',
             'cohort_prior4w','anchor_ly_pop','spend_ly4w','has_ly4w','pop_ratio']
disc_cols = ['disc_net_84','coupon_redemptions_84','dsl_coupon']
cal_cols  = ['wk_sin','wk_cos','wk_sin2','wk_cos2']

add = e14[['household_key','snapshot_day']+demo_cols+l13_cols]
add = add.merge(e13[['household_key','snapshot_day']+peer_cols], on=['household_key','snapshot_day'], how='left')
add = add.merge(v2[['household_key','snapshot_day']+disc_cols+cal_cols], on=['household_key','snapshot_day'], how='left')

full = e16.merge(add, on=['household_key','snapshot_day'], how='left')
print(full.shape, "dups:", full.duplicated(['household_key','snapshot_day']).sum())
print("demo coverage:", full['has_demographics'].mean())
print("peer coverage:", full['peer_recent28'].notna().mean())
print("wk_sin coverage:", full['wk_sin'].notna().mean())
path = api.save_table(full, "e019_everything")
print(path)

# ---- cell ----
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