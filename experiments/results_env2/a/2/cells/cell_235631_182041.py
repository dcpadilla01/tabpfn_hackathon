import agent_api as A, pandas as pd, numpy as np, itertools

tt = A.train_targets()
oof = A.load_saved('oof_e008.parquet').merge(tt, on=['household_key','snapshot_day'])
y = oof.future_spend_4w.values
P = {c: oof[c].values for c in ['oof_sq','oof_med','oof_log']}
print('E006-style blend MAE:', np.abs(0.5*P['oof_sq']+0.5*P['oof_med']-y).mean())
print('med only:', np.abs(P['oof_med']-y).mean())

# grid-search blend weights (step .05, sum=1)
best=(1e9,None)
for a in np.arange(0,1.01,0.05):
    for b in np.arange(0,1.01-a,0.05):
        c = 1-a-b
        p = a*P['oof_sq']+b*P['oof_med']+c*P['oof_log']
        m = np.abs(p-y).mean()
        if m<best[0]: best=(m,(round(a,2),round(b,2),round(c,2)))
print('best 3-blend:', best)

# clip test on med and on best blend
for cap in [150,200,250,300,400,500,1e9]:
    p = np.clip(P['oof_med'],0,cap)
    print('med clip',cap, round(np.abs(p-y).mean(),3))

# per-household shrink: blend pred with household's OOF median spend? need household history; use per-household mean of y? not available at pred time legitimately... skip
# check feats_v3 cols
f3 = A.load_saved('feats_v3.parquet')
print([c for c in f3.columns if 'snap' in c or 'week' in c or 'day' in c])
