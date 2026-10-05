import pandas as pd, numpy as np

names = ["churn_vol_v1","combined_v1","comp_v1","demo_v1","ewma_block_v1","mkt_v1","rfm_cadence_v1","rfm_traj_v1","rfm_v1","season_demo_v1","union_all_v1"]
tabs = {}
for n in names:
    df = load_saved(n + ".parquet")
    tabs[n] = df
    print(n, df.shape)
    print("  ", list(df.columns))

tt = train_targets()
print("\ntargets:", tt.shape)
print(tt.future_spend_4w.describe())
print("\nmean target by snapshot_day:")
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))
