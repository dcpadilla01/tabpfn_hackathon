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