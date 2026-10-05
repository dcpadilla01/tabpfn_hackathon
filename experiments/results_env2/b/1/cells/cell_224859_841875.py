
import pandas as pd, numpy as np

names = ["e001_history","e002_mix","e003_full","e003_momentum","e004_marketing","e006_cadence","e007_temporal","e008_level_shape","e009_target_enc","e010_gross"]
for n in names:
    df = load_saved(n + ".parquet")
    print("==", n, df.shape)
    print(sorted([c for c in df.columns if c not in ("household_key","snapshot_day")]))

t = train_targets()
print("\ntargets", t.shape)
print(t[TARGET].describe())
print("zero frac:", (t[TARGET]==0).mean())
print("snapdays:", snapshot_days())
