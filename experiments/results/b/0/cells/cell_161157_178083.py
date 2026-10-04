import agent_api as A
import pandas as pd, numpy as np

names = ["rfm28","rich_behavioral","season","macro","mkt_demo","e006_composition","e007_lagseq",
         "e010_decay","e011_price","e009_basket","dm_exp","e016_peer","e013_te_clean","cand1","e014_dm","e012_xenc"]
for n in names:
    t = A.load_saved(n + ".parquet")
    cols = list(t.columns)
    print(f"== {n} {t.shape}")
    print("   " + ", ".join(cols)[:1400])

print("\nsnapshot_days:", A.snapshot_days())
base = A.baseline_features()
print("baseline rows:", len(base))

tt = A.train_targets()
print("targets:", tt.shape)
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))
print(tt["future_spend_4w"].describe().round(2))
print("zero frac:", round(float((tt.future_spend_4w==0).mean()),4))
