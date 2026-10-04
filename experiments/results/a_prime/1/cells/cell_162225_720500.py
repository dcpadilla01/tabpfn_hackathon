import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 250)
for name in ['e016_smoothed.parquet','e013_peers.parquet','micro.parquet','nf_candidates.parquet']:
    df = agent_api.load_saved(name)
    print('==', name, df.shape)
    print(list(df.columns))
    print()
t = agent_api.train_targets()
print('targets', t.shape)
print(t['future_spend_4w'].describe())
v = agent_api.snapshot()
tx = v.transactions
print('tx rows', len(tx), 'max day', tx['day'].max())
print(tx[['retail_disc','coupon_disc','coupon_match_disc','quantity','sales_value']].describe().round(2))
# population weekly spend per active household (seasonality check)
fd = tx.groupby('household_key')['day'].min()
tx2 = tx.copy(); tx2['wk'] = (tx2['day']+8)//7
wk_tot = tx2.groupby('wk')['sales_value'].sum()
base_n = pd.Series({w: (fd <= 7*w-8).sum() for w in wk_tot.index})
phh = (wk_tot/base_n)
print('weekly per-household spend, last 58 weeks:')
print(phh.tail(58).round(1).to_string())
print('overall mean', round(phh.mean(),1), 'cv', round(phh.std()/phh.mean(),3))