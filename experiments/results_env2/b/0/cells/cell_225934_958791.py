import pandas as pd, agent_api
fe = agent_api.load_saved("e006_zero_inflation.parquet")
drop = ["lines_84","lines_per_trip","avg_basket_84","avg_basket_84_ts","unit_price_84",
        "qty_28","qty_84","national_share_84","n_products_84","n_depts_84",
        "evening_share_84","weekend_share_84","disc_share_84"]
drop = [c for c in drop if c in fe.columns]
out = fe.drop(columns=drop)
print("kept:", out.shape[1]-2, "dropped:", drop)
p = agent_api.save_table(out, "e011_pruned_basket.parquet")
print(p)
