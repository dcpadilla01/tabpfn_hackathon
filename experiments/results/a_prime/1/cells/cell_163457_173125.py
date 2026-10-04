import agent_api as api

# What's in the current best table and the prepared-but-unused tables?
for name in ["e016_smoothed", "e017_v2", "e017_disc_seasonal", "e018_timing_hazard"]:
    df = api.load_saved(name + ".parquet")
    print("==", name, df.shape)
    print(list(df.columns))
    print()