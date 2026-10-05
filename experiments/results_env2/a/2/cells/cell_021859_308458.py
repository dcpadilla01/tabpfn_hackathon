import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med","oof_sq","oof_med_w112"]], on=["household_key","snapshot_day"])
y = o.future_spend_4w.values
cols = ["oof_med","oof_sq","oof_log","oof_med","oof_sq","oof_med_w112"]
cols[3] = "oof_med_2"; cols[4]="oof_sq_2"; o["oof_med_2"]=o.oof_med_y; o["oof_sq_2"]=o.oof_sq_y
cols = ["oof_med_x","oof_sq_x","oof_log","oof_med_2","oof_sq_2","oof_med_w112"]
o = o.rename(columns={"oof_med_x":"oof_med_x","oof_sq_x":"oof_sq_x"})
# figure out actual col names
print([c for c in o.columns])
