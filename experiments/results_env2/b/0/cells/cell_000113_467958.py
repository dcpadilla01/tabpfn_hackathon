import agent_api, pandas as pd, numpy as np
pd.set_option("display.width", 250)

t = agent_api.load_saved("e013_stationary.parquet")
print("E013 shape:", t.shape)
print("E013 cols:", list(t.columns))

tt = agent_api.train_targets()
print("\nTarget by train snapshot:")
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby("week_no").agg(spend=("sales_value","sum"), baskets=("basket_id","nunique"), hhs=("household_key","nunique"))
g["spend_per_basket"] = g.spend/g.baskets
print("\nWeekly aggregates (week_no, spend, baskets, hhs, spend/basket):")
print(g.round(1).to_string())
