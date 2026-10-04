import agent_api as A, pandas as pd, numpy as np

e11 = A.load_saved("e011_demo.parquet")
tt = A.train_targets()
m = tt.merge(e11[["household_key","snapshot_day","spend_l1","spend_l2","spend_rate28","zero_recent","tenure"]],
             on=["household_key","snapshot_day"])

micro = A.load_saved("micro.parquet")[["household_key","snapshot_day","mspend_l1","spend_7d","spend_14d","trips_14d"]]
m = m.merge(micro, on=["household_key","snapshot_day"], how="left")
print("corr mspend_l1 vs spend_l1:", m["mspend_l1"].corr(m["spend_l1"]))
print(m[["spend_l1","mspend_l1","spend_7d","spend_14d","trips_14d"]].describe().round(2))

peers = A.load_saved("e013_peers.parquet")[["household_key","snapshot_day","peer_ratio","peer_recent28","spend_ly4w","nratio_ly" if "nratio_ly" in A.load_saved("e013_peers.parquet").columns else "peer_p90"]]
m = m.merge(peers, on=["household_key","snapshot_day"], how="left")
print("\npeer_ratio describe:"); print(m["peer_ratio"].describe().round(3))
print("corr peer_recent28 vs spend_l1:", m["peer_recent28"].corr(m["spend_l1"]))
print(m[["peer_ratio","peer_recent28","spend_ly4w"]].head(8).round(3))

seas = A.load_saved("nf_seasonal.parquet")
m = m.merge(seas, on=["household_key","snapshot_day"], how="left")
print("\nnratio_ly describe:"); print(m["nratio_ly"].describe().round(3))
print(m[["spend_l1","nratio_ly","nspend_ly","nseas_uplift"]].head(8).round(3))
