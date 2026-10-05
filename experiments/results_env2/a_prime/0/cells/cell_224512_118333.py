import agent_api, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
df = agent_api.load_saved('e009_ewma.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = d[d.snapshot_day <= 403]; va = d[d.snapshot_day == 431]
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values

# inner-val check of the new ew features vs spend_112w
for c in ['ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ew_rate_hl28','ew_log_hl112','ew_bask_hl28']:
    x = tr[c]; ok = x.notna()
    qs = np.quantile(x[ok], np.linspace(0,1,21)); qs[0]-=1; qs[-1]+=1
    b = np.digitize(x[ok], qs)
    m = pd.Series(ytr[ok.values]).groupby(b).mean()
    pv = pd.Series(np.digitize(va[c].fillna(tr[c].median()).values, qs)).map(m).fillna(ytr.mean()).values
    r = np.corrcoef(x[ok], ytr[ok.values])[0,1]
    print(f'{c:16s} innerMAE {np.mean(np.abs(pv-yva)):6.2f} corr {r:+.3f}')

# residual target: what remains after best-known single predictor spend_112w
def resid_target(tr, va, col='spend_112w'):
    x = tr[col]; ok = x.notna()
    qs = np.quantile(x[ok], np.linspace(0,1,21)); qs[0]-=1; qs[-1]+=1
    b = np.digitize(x[ok], qs)
    m = pd.Series(ytr[ok.values]).groupby(b).mean()
    pr = pd.Series(np.digitize(va[col].fillna(tr[col].median()).values, qs)).map(m).fillna(ytr.mean()).values
    return pr

pr = resid_target(tr, va)
print('spend_112w alone inner MAE:', round(np.mean(np.abs(pr-yva)),2))
resid = yva - pr
# which features correlate with the residual?
feat = [c for c in d.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
rows=[]
for c in feat:
    x = tr[c].values
    if not np.issubdtype(tr[c].dtype, np.number): continue
    ok = ~np.isnan(x)
    if ok.sum()<100: continue
    r = np.corrcoef(x[ok], (ytr - resid_target(tr, tr))[ok])[0,1]
    rows.append((c, r))
rows.sort(key=lambda t: -abs(t[1]))
print('\nFeatures most correlated with residual (after spend_112w):')
for c,r in rows[:20]: print(f'{c:28s} {r:+.3f}')
