import pandas as pd, numpy as np
import agent_api as A

snap = A.snapshot(459)
tx = snap.transactions
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
print("macro weekly per-hh spend, weeks 1..66:")
print(macro_w.round(1).to_string())