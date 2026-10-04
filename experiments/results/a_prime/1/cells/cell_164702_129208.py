for name in ["e017_disc_seasonal.parquet","e019_everything.parquet"]:
    df = load_saved(name)
    print("===", name, df.shape)
    print(list(df.columns))
    print()