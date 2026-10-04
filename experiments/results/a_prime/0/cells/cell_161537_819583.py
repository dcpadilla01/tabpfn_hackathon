import agent_api, pandas as pd, numpy as np
for name in ["e013_stock.parquet","e011_rank.parquet","stock_v1.parquet","rhythm_v1.parquet","mkt_v2.parquet","e012_embed.parquet"]:
    df = agent_api.load_saved(name)
    print("==", name, df.shape)
    print(list(df.columns))
print()
v = agent_api.snapshot()
dm = v.display_mailer
print("display_mailer", dm.shape)
print(dm.head(8))
print("display uniq:", dm.display.unique()[:20])
print("mailer uniq:", dm.mailer.unique()[:20])
cr = v.coupon_redemptions
print("coupon_redemptions", cr.shape); print(cr.head())
cp = v.coupons
print("coupons", cp.shape); print(cp.head())
t = v.transactions
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity','trans_time']].describe())
print(agent_api.snapshot_days())
