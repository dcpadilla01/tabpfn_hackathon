import pandas as pd, numpy as np
from agent_api import load_saved, save_table
KEY = ["household_key","snapshot_day"]
base = load_saved("e013_union.parquet")
seq = load_saved("e006_seq_gaps.parquet")
A = ["ew28","ew56","weekly_rate_84","w_max7","w_min7","w_std7","w_cv7","spike","gap_max182","gap_med",
     "nb_all","n_gap21_182","n_gap14_84"] + [f"w{i}" for i in range(1,14)]
dem = ["has_demographics","classification_1","classification_3","classification_4","classification_5",
       "homeowner_desc","kid_category_desc"]
want = A + dem
Bd = [c for c in want if c in seq.columns]
print("missing:", [c for c in want if c not in seq.columns])
tab = base.merge(seq[KEY+Bd], on=KEY, how="left")
print("shape:", tab.shape, "| new feats:", len(Bd))
path = save_table(tab, "e015_base")
print("saved:", path)
