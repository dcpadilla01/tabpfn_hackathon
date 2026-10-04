import agent_api as A

names = ["e011_demo.parquet","e012_dorm.parquet","e013_peers.parquet",
         "micro.parquet","nf_candidates.parquet","nf_compact19.parquet",
         "nf_p1.parquet","nf_robust.parquet","nf_seasonal.parquet","nf_transforms.parquet"]
for n in names:
    try:
        df = A.load_saved(n)
        print("==", n, df.shape)
        print(list(df.columns))
    except Exception as e:
        print("==", n, "ERR", repr(e))
