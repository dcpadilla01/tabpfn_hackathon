
import pandas as pd, numpy as np

tt = train_targets()
print("train target rows:", tt.shape)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","std","count"])
print(g.round(2))

t17 = load_saved("e017_merged.parquet")
cols = list(t17.columns)
import re
# find RFM-ish spend columns to reuse for relative features
cand = [c for c in cols if re.search(r"(spend|sp_|s28|s56|ewma|ma4|mean)", c, re.I)]
print(len(cand))
print(sorted(cand)[:60])
print([c for c in cols if "tenure" in c or "day" in c][:20])

# departments/commodities for possible later use
snap = snapshot()
tx = snap.transactions
print("tx shape", tx.shape)
print("n departments", tx.product_id.map(dict(zip(snap.products.product_id, snap.products.department))).nunique() if hasattr(snap,'products') else "n/a")
