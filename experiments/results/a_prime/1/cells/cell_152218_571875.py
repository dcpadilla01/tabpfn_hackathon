import agent_api as A
import pandas as pd, numpy as np

print("== snapshot days ==")
print(A.snapshot_days())

tt = A.train_targets()
y = tt.future_spend_4w
print("== target train ==")
print(y.describe())
print("zero share:", round(float((y==0).mean()),4), " median:", float(y.median()))

names = ["e001_txhist","e002_channel","e003_catmix","e004_mkt","e004_mkt_full","e006_catmix_mkt","e007_logratio","nf_candidates","nf_p1","nf_seasonal","nf_transforms"]
tabs = {}
for nm in names:
    try:
        df = A.load_saved(nm + ".parquet")
        tabs[nm] = df
        fcols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
        print(f"{nm}: shape={df.shape} nfeat={len(fcols)}")
    except Exception as e:
        print(nm, "ERR", repr(e))

print("\nE003 cols:", list(tabs["e003_catmix"].columns))
print("\nE001 cols:", list(tabs["e001_txhist"].columns))
print("\nnf_candidates cols:", list(tabs["nf_candidates"].columns))

v = A.snapshot()
tx = v.transactions
print("\ntransactions shape:", tx.shape)
print(tx.head(3))
print("discount signs (means):", tx[["coupon_disc","coupon_match_disc","retail_disc","sales_value","quantity"]].mean())
