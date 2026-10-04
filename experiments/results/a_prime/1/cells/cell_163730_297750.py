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