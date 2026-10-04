import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
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
def wridge_fit(Xtr, ytr, wt, Xva, alpha=100.0):
    mu = (Xtr*wt[:,None]).sum(0)/wt.sum(); sg = np.sqrt(((Xtr-mu)**2*wt[:,None]).sum(0)/wt.sum())+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=(ytr*wt).sum()/wt.sum()
    ZtW = Z*wt[:,None]
    w = np.linalg.solve(Z.T@ZtW+alpha*np.eye(Z.shape[1]), ZtW.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med): return df[cols].astype(float).fillna(med).values

# market aggregates per snapshot (from D rows at that snapshot = observable at s)
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)  # snapshot s-28's trailing-28d market mean = window [s-56, s-28]
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
D['drift_x'] = D['drift28']*np.log1p(D['spend_28'].astype(float))
mcols = ['mkt28','mkt_mom','drift28','drift_x']
mm = D[mcols].median()
print('mkt_mom by snap:', D.groupby('snapshot_day')['mkt_mom'].first().round(3).to_dict())

def run_variant(add_cols, weighted=None, snaps_test=[375,403,431], alpha=100.0):
    maes=[]
    for s in snaps_test:
        ptr = D[(D.snapshot_day.isin(trd)) & (D.snapshot_day < s)]
        pv = D[D.snapshot_day==s]
        cols = feat0 + [c for c in add_cols if c in D.columns]
        med = ptr[cols].astype(float).median()
        Xtr = build_X(ptr, cols, med); Xpv = build_X(pv, cols, med)
        ytr = ptr[TARGET].values; ypv = pv[TARGET].values
        if weighted:
            wt = 0.5**((s-ptr['snapshot_day'].values)/weighted)
            p = wridge_fit(Xtr,ytr,wt,Xpv,alpha)
        else:
            p = ridge_fit(Xtr,ytr,Xpv,alpha)
        maes.append(np.abs(p-ypv).mean())
    return maes

for name, add, w in [('base',[],None), ('base+w150',[],150), ('drift',['mkt28','mkt_mom','drift28'],None),
                     ('drift+w150',['mkt28','mkt_mom','drift28'],150), ('drift+dx',['mkt28','mkt_mom','drift28','drift_x'],None),
                     ('drift+w150+dx',['mkt28','mkt_mom','drift28','drift_x'],150),
                     ('drift+w150+dx+shrink',['mkt28','mkt_mom','drift28','drift_x'],150)]:
    if 'shrink' in name:
        k=16; gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
        D['shr'] = (D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k))*D['spend_28'].astype(float) + (1-D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k))*gmed
        add = add+['shr']
    m = run_variant(add, w)
    print(f'{name:24s} MAE per snap {[round(x,2) for x in m]} avg {np.mean(m):.3f}')