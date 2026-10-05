import agent_api as A, pandas as pd
f3 = A.load_saved("feats_v3.parquet"); f5 = A.load_saved("feats_v5.parquet")
print(f3.shape, f5.shape)
print("f5 dup labels:", f5.columns[f5.columns.duplicated()].tolist())
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
print("overlap with v3:", [c for c in NEW if c in f3.columns])
print([c for c in f5.columns if "y1" in c or "y2" in c])
