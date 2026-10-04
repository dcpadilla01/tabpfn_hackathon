import agent_api as A
import pandas as pd, numpy as np

names = ["e015_base","e013_union","e009_macro","e006_seq_gaps","cand_new","e001_history","e002_marketing",
         "e003_dept_mix","e004_long_hist","e005_seasonal_peer","e007_new","e008_decomp2","e010_composite",
         "e011_display","e012_full","e014_base"]
tabs = {}
for nm in names:
    df = A.load_saved(nm + ".parquet")
    tabs[nm] = df
    print(f"{nm}: shape={df.shape}")

base = tabs["e015_base"]
bcols = set(base.columns)
print("\n--- extra columns vs e015_base ---")
for nm in names:
    extra = [c for c in tabs[nm].columns if c not in bcols]
    print(f"{nm}: n_extra={len(extra)}")
    if 0 < len(extra) <= 30:
        print("   ", extra)

bl = A.baseline_features()
print("\nbaseline_features cols:", list(bl.columns))
print("baseline shape:", bl.shape)

tt = A.train_targets()
print("\ntrain_targets:", tt.shape)
print(tt.future_spend_4w.describe())
print("\nshare zeros:", (tt.future_spend_4w==0).mean())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))