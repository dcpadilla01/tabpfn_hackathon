import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
m = tt.merge(f3, on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
# check spend_28 == 0 rows: how many, and what's y distribution
z = m[m.spend_28==0]
print("n spend28==0:", len(z), "P(y==0):", round((z.future_spend_4w==0).mean(),3), "y>0 mean:", round(z[z.future_spend_4w>0].future_spend_4w.mean(),1))
# among spend28==0, does days_since_last matter?
for lo,hi in [(0,28),(28,56),(56,84),(84,10**9)]:
    s = z[(z.days_since_last>=lo)&(z.days_since_last<hi)]
    print(f"dsl[{lo},{hi}) n={len(s)} P(y==0)={round((s.future_spend_4w==0).mean(),2)} y>0 mean={round(s[s.future_spend_4w>0].future_spend_4w.mean(),1)}")
