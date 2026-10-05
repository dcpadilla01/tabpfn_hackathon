import pandas as pd
df = agent_api.load_saved("e007_te.parquet")
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
print(len(feats))
for i in range(0, len(feats), 8):
    print(" | ".join(f"{c:28s}" for c in feats[i:i+8]))
