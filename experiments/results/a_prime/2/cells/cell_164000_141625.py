e009 = agent_api.load_saved("e009_ewma_longlags.parquet")
e9cols = set(e009.columns)
for p in ["churn_seasonality.parquet","behavioral_candidates.parquet","dept_mix_recency.parquet","e013_new_feats.parquet","e012_robust.parquet"]:
    df = agent_api.load_saved(p)
    new = [c for c in df.columns if c not in e9cols]
    print("==", p, df.shape, "| new vs E009:", len(new))
    print(new)
    print()