import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

# NaN structure
nanrate = e008.isna().mean().sort_values(ascending=False)
print('cols with NaN>0:'); print((nanrate[nanrate>0]*100).round(1).to_string())

def fwd_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(B@w - yva).mean()

# 1) day x level interactions
d = e008.copy()
for lev in ['spend_84','spend_28','L_s84','L_s28']:
    if lev in d.columns:
        d['di_x_'+lev] = d['day_idx']*d[lev]
    else:
        # L_s84 not in e008; approximate with log of spend_84
        d['L_s84'] = np.log1p(d['spend_84'].clip(lower=0)); d['L_s28'] = np.log1p(d['spend_28'].clip(lower=0))
        d['di_x_'+lev] = d['day_idx']*d[lev]
print('\n[1] day x level interactions:')
print('  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('  +dayxlevel  v431: %.3f' % fwd_mae(d,[431]))

# 2) marketing under forward proxies
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('\n[2] marketing block:')
print('  E008      v431: %.3f | v403: %.3f' % (fwd_mae(e008,[431]), fwd_mae(e008,[403])))
print('  +mkt      v431: %.3f | v403: %.3f' % (fwd_mae(c02,[431]), fwd_mae(c02,[403])))

# 4) winsorize heavy spend cols at per-snapshot p99
w = e008.copy()
for c in ['spend_84','spend_112','spend_56','spend_168','spend_28','spend_364','spend_life','fwd28_max','fwd28_mean','wk_avg_84','wk_avg_8']:
    if c in w.columns:
        cap = w.groupby('snapshot_day')[c].transform(lambda s: s.quantile(0.99))
        w[c+'_w'] = np.minimum(w[c].fillna(0), cap)
print('\n[4] winsorized top spends:')
print('  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('  +winsor     v431: %.3f' % fwd_mae(w,[431]))

# 5) median imputation variant (proxy only)
def fwd_mae_med(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float); med = Xtr.median(); Xtr = Xtr.fillna(med).values
    Xva = va[f].astype(float).fillna(med).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    ww = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(B@ww - yva).mean()
print('\n[5] imputation: zero vs median (E008, v431): %.3f vs %.3f' % (fwd_mae(e008,[431]), fwd_mae_med(e008,[431])))