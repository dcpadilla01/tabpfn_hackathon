import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
print(tt.shape, tt.columns.tolist())

oof8 = A.load_saved("oof_e008.parquet")
print(oof8.shape, oof8.columns.tolist())
m = tt.merge(oof8, on=["household_key","snapshot_day"], how="inner")
print("merged", m.shape)

def mae(p, y): return np.mean(np.abs(p-y))
y = m.future_spend_4w.values
for c in ["oof_sq","oof_med","oof_log"]:
    print(c, round(mae(m[c].values, y),3))

# grid search blend weights over the three oof components
best=(1e9,None)
import itertools
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3=1-w1-w2
        p=w1*m.oof_med.values+w2*m.oof_sq.values+w3*m.oof_log.values
        v=mae(p,y)
        if v<best[0]: best=(v,(round(w1,2),round(w2,2),round(w3,2)))
print("best blend oof:", best)
