for name in ["e017_v2.parquet","e018_timing_hazard.parquet","e019_everything.parquet","e019_full_merged.parquet"]:
    df = load_saved(name)
    print("=== ", name, df.shape)
    print(list(df.columns))
    print()