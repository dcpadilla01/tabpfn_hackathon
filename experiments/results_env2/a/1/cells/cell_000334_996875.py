import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
va = F[F.snapshot_day==431]
yv = va.future_spend_4w.values
# heuristics
for name, p in [('spend_28', va.spend_28.values), ('spend_84', va.spend_84.values),
                ('(28+84)/2', (va.spend_28.values+va.spend_84.values)/2),
                ('spend_56', va.spend_56.values)]:
    print('%-10s MAE %.3f' % (name, np.abs(p-yv).mean()))
print('E005-config xgb internal: 63.049 (ref)')

# build lag features via fresh build_features (only need lag windows; cheap fn)
def lag_fn(view, day):
    tx = view.table('transactions')
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    out = pd.DataFrame(index=hh)
    for lo, hi, nm in [(56,28,'lag28_56'), (84,56,'lag56_84'), (112,84,'lag84_112'), (140,112,'lag112_140')]:
        w = g[(g.day > day-hi) & (g.day <= day-lo)].groupby('household_key').sales_value.sum()
        out[nm] = w.reindex(hh).fillna(0.0)
    return out

t0=time.time()
L = agent_api.build_features(lag_fn)
print('lag feats built %.0fs' % (time.time()-t0), L.shape)
path = agent_api.save_table(L.reset_index(), 'lagfeats.parquet')
print(path)
