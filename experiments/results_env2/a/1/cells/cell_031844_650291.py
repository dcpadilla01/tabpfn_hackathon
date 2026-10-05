
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], how="inner", suffixes=('_F','_T'))
print("merged:", m.shape)
print(m[['future_spend_4w_F','future_spend_4w_T']].head())
print("equal:", np.allclose(m.future_spend_4w_F, m.future_spend_4w_T))
t = m['future_spend_4w_T']
g = m.groupby("snapshot_day").apply(lambda d: pd.Series({'mean':d.future_spend_4w_T.mean(),'med':d.future_spend_4w_T.median(),'zero':(d.future_spend_4w_T==0).mean()}))
print(g.round(2))
m['wk'] = ((m.snapshot_day+8)//7) % 52
print(m.groupby('wk')['future_spend_4w_T'].mean().round(1).to_string())
