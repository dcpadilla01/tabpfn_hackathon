
import numpy as np, pandas as pd

tt = agent_api.train_targets()
y = tt['future_spend_4w']
print("targets:", tt.shape, "zero share:", round(float((y==0).mean()),3))
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))

e012 = agent_api.load_saved('e012_style.parquet')
print("\ne012:", e012.shape)
feats = [c for c in e012.columns if c not in ('household_key','snapshot_day')]
print("n_feats:", len(feats))
print("dtypes:", e012[feats].dtypes.value_counts().to_dict())
nonnum = [c for c in feats if not np.issubdtype(e012[c].dtype, np.number)]
print("non-numeric:", nonnum)
print("NaN cols>50%:", [c for c in feats if e012[c].isna().mean()>0.5][:10])

e015 = agent_api.load_saved('e015_stack.parquet')
print("\ne015 extra cols:", [c for c in e015.columns if c not in e012.columns])
print("snapshots:", sorted(e012.snapshot_day.unique()))
print("rows per snapshot:", e012.groupby('snapshot_day').size().to_dict())
