
import agent_api, numpy as np, pandas as pd
names = ["e005_preds","e011_preds","e004_preds","e003_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds"]
preds = {}
for n in names:
    d = agent_api.load_saved(n + ".parquet")
    preds[n] = d.set_index(["household_key","snapshot_day"])['prediction']
P = pd.DataFrame(preds)
print(P.shape)
C = P.corr()
print(C.round(4).to_string())
print("\nmean pred per table:", P.mean().round(2).to_string())
print("\nstd per table:", P.std().round(2).to_string())
# equal-weight blend of the three best
b3 = P[['e005_preds','e011_preds','e004_preds']].mean(axis=1)
print("\nblend3 vs e005: corr", np.corrcoef(b3, P['e005_preds'])[0,1].round(5), "mean", b3.mean().round(2))
