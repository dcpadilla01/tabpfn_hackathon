import agent_api as A
for name in ["behavioral_candidates","churn_seasonality","dept_mix_recency","marketing_exposure"]:
    df = A.load_saved(name + ".parquet")
    print(name, df.shape)
    print(list(df.columns))
    print("---")