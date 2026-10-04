import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days, save_table
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = sorted(snapshot_days()['train'])
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
def ridge_fit(Xtr, ytr, Xva, alpha=100.0):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med): return df[cols].astype(float).fillna(med).values

# market features from saved table (within-snapshot aggregates of observable features)
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
k=16; w_ = D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k)
D['shr'] = w_*D['spend_28'].astype(float) + (1-w_)*gmed
newcols = ['mkt28','mkt_mom','drift28','shr']
print(D[newcols].describe().round(3))

def run_variant(add_cols, snaps_test, alpha):
    maes=[]
    for s in snaps_test:
        ptr = D[(D.snapshot_day.isin(trd)) & (D.snapshot_day < s)]; pv = D[D.snapshot_day==s]
        cols = feat0 + add_cols
        med = ptr[cols].astype(float).median()
        p = ridge_fit(build_X(ptr,cols,med), ptr[TARGET].values, build_X(pv,cols,med), alpha)
        maes.append(np.abs(p-pv[TARGET].values).mean())
    return np.mean(maes), [round(x,2) for x in maes]
snaps_test=[347,375,403,431]
for name, add in [('base',[]), ('+mkt4',newcols)]:
    for a in [30,100,300]:
        avg, per = run_variant(add, snaps_test, a)
        print(f'{name:8s} alpha={a:4d}: avg {avg:.3f} per {per}')
# save final candidate table (all 17 snapshot rows)
OUT = F[key+feat0].copy()
for c in newcols: OUT[c] = D[c].values
OUT['mkt_mom'] = OUT['mkt_mom'].fillna(OUT['mkt_mom'].median())
path = save_table(OUT, 'e015_market_ctx')
print('saved:', path, OUT.shape)
print('sanity: rows per snap', OUT.groupby('snapshot_day').size().to_dict())