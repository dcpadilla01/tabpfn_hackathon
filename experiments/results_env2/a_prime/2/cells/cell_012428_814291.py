
import agent_api as A
t18 = A.load_saved("e018_tree_feats.parquet")
t17 = A.load_saved("e017_xsec_rank.parquet")
print("E018 cols:", list(t18.columns))
print("E018 shape:", t18.shape)
print("E017 shape:", t17.shape)
print("E017 cols:", list(t17.columns))
print("E018 snapshot days:", sorted(t18.snapshot_day.unique()))
print("E017 snapshot days:", sorted(t17.snapshot_day.unique()))
