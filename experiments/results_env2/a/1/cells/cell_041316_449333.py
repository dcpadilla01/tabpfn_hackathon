import agent_api as api, pandas as pd, numpy as np
allF = api.load_saved('allF.parquet')
feats = ['spend_28','spend_84','spend_364','actdays_28','recency','bask_28','spend_total','tenure']
print("Feature means by snapshot day (drift check):")
print(allF.groupby('snapshot_day')[feats].mean().round(1))
# household overlap train vs val
tr_h = set(allF[allF.snapshot_day<=431].household_key); va_h = set(allF[allF.snapshot_day>=459].household_key)
print("train hh:", len(tr_h), "val hh:", len(va_h), "val not in train:", len(va_h-tr_h))
