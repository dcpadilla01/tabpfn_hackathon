import agent_api, numpy as np, pandas as pd

df = agent_api.load_saved('e007_te.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='inner')

# check te_prior: constant or per-snapshot?
print(d.groupby('snapshot_day')['te_prior'].agg(['mean','std','min','max']).head(15))

# inner split: fit on snapshots 95..403, eval on 431 (proxy for val)
tr = d[d.snapshot_day <= 403]; va = d[d.snapshot_day == 431]
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
print('inner val rows', len(va), 'mean', round(yva.mean(),1))

def mae(p, y): return np.mean(np.abs(p-y))
print('baseline mean-pred MAE:', round(mae(np.full(len(yva), ytr.mean()), yva),2))

# univariate: predict by mapping feature value -> train mean target in quantile bins (fit inner-train)
feat = [c for c in d.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
res = []
for c in feat:
    x = tr[c]
    if not np.issubdtype(x.dtype, np.number): continue
    ok = x.notna()
    if ok.sum() < 100: continue
    try:
        qs = np.quantile(x[ok], np.linspace(0,1,21))
        qs[0]-=1; qs[-1]+=1
        b = np.digitize(x[ok], qs)
        m = pd.Series(ytr[ok.values]).groupby(b).mean()
        p = pd.Series(b, index=tr.index[ok.values]).map(m)
        pr = pd.Series(np.nan, index=tr.index); pr[tr.index[ok.values]] = p.values
        pr = pr.fillna(ytr.mean())
        # map to inner-val
        bx = np.digitize(va[c].fillna(tr[c].median()).values, qs)
        pv = pd.Series(bx).map(m).fillna(ytr.mean()).values
        r = np.corrcoef(x[ok], ytr[ok.values])[0,1] if x[ok].std()>0 else 0
        res.append((c, mae(pv, yva), r))
    except Exception as e:
        pass
res.sort(key=lambda t: t[1])
print('\nTop 25 by univariate inner-val MAE (overall mean-pred MAE shown above):')
for c,m,r in res[:25]: print(f'{c:28s} {m:7.2f} corr {r:+.3f}')
