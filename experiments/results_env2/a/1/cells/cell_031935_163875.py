
import agent_api, numpy as np, pandas as pd
P = {}
for n in ["e005_preds","e011_preds","e004_preds","e003_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds"]:
    P[n] = agent_api.load_saved(n + ".parquet").set_index(["household_key","snapshot_day"])['prediction']
P = pd.DataFrame(P)
# per snapshot-day mean prediction, val days
v = P.reset_index()
g = v.groupby('snapshot_day').agg(e005=('e005_preds','mean'), e011=('e011_preds','mean'),
                                  e004=('e004_preds','mean'), e003=('e003_preds','mean'),
                                  e008=('e008_preds','mean'), e009=('e009_preds','mean'))
print(g.round(2).to_string())
# what does e005 say vs e008 (trained incl 431)? e008 mean 139.76 vs e005 127.10
# Train-side target means by day: 403:146.5, 431:141.0. Later train days higher -> e008 higher preds.
# Check e005's per-day MAE pattern can't be computed (no val targets). Use train-side surrogate:
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
print("\ntrain target mean by day:\n", m.groupby('snapshot_day')['future_spend_4w_T'].mean().round(1).to_string())
