import agent_api as A, pandas as pd
for name in ["basket_tenure","behavioral_candidates","churn_seasonality","dept_mix_recency","marketing_exposure","recency_agg","temporal_structure"]:
    try:
        df = A.load_saved(name + ".parquet")
        print(name, df.shape)
        print(list(df.columns))
        print("---")
    except Exception as e:
        print(name, "ERR", e)
