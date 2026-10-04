paths = ["churn_seasonality.parquet","behavioral_candidates.parquet","dept_mix_recency.parquet","e013_new_feats.parquet","e012_robust.parquet","e009_ewma_longlags.parquet"]
for p in paths:
    df = agent_api.load_saved(p)
    print("==", p, df.shape)
    cols = list(df.columns)
    print(len(cols), "cols")
    print(cols)
    print("---")