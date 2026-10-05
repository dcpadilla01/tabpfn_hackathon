
import agent_api as api, numpy as np, pandas as pd

v = api.snapshot(459)
t = v.transactions
wk = t.groupby("week_no").sales_value.sum()
print("weeks range:", wk.index.min(), wk.index.max(), "n weeks:", len(wk))
# overall weekly mean per week index
m = wk.mean(); s = wk.std()
print("mean weekly total spend:", round(m), "std:", round(s))
big = wk[wk > m + 2*s]
print("\nSpike weeks (> mean+2sd):")
print(big.round(0))
print("\nTop 12 weeks by spend:")
print(wk.sort_values(ascending=False).head(12).round(0))
print("\nBottom 5 weeks:")
print(wk.sort_values().head(5).round(0))
# also transactions count per week
cnt = t.groupby("week_no").size()
print("\nTop 6 weeks by trip count:")
print(cnt.sort_values(ascending=False).head(6))
# what day range does week w cover: week_no=(day+8)//7 -> day in [7w-8, 7w-2]
print("\nweek->day mapping examples: week 66 -> days", 7*66-8, "to", 7*66-2)
