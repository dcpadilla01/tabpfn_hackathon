import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
day=403
v = agent_api.snapshot(as_of_day=day); t = v.transactions
out = pd.DataFrame(index=v.households)
for k in range(1,8):
    lo = day-28*k
    out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
out = out.reset_index()
print('reset_index cols:', out.columns.tolist(), 'index name was:', v.households.name if hasattr(v.households,'name') else type(v.households))
print(out.head(3))
out['y'] = out['household_key'].map(tt[tt.snapshot_day==day].set_index('household_key').future_spend_4w)
print('y notna frac:', round(out.y.notna().mean(),3), 'n rows', len(out))
L = np.log1p(out[['w1','w2','w3','w4','w5','w6','w7']].fillna(0).clip(lower=0))
Ly = np.log1p(out.y.fillna(0))
print('corr(log w, log y):'); print(L.corrwith(Ly).round(3))
print('corr raw w1 vs y:', round(out.w1.fillna(0).corr(out.y.fillna(0)),3))
# check tt day 403 coverage
print('tt day403 n:', (tt.snapshot_day==403).sum())