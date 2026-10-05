import agent_api, numpy as np, pandas as pd

df = agent_api.load_saved('e007_te.parquet')
print('shape', df.shape)
print(df.dtypes.value_counts())
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', d.shape)
y = d['future_spend_4w'].values
print('target mean/med/zero-frac (train):', 
      round(np.mean(y),1), round(np.median(y),1), round(np.mean(y==0),3))

feat = [c for c in d.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n feat', len(feat))
print('columns:', feat)


# ---- cell ----
import agent_api, numpy as np, pandas as pd
df = agent_api.load_saved('e008_spendproc.parquet')
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('E008 n feat', len(feat))
new = [c for c in feat if c not in set(agent_api.load_saved('e007_te.parquet').columns)]
print('E008 added vs E007:', new)


# ---- cell ----
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


# ---- cell ----
import agent_api, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

# Candidate: recency-weighted (exponential decay) spend/basket features
def fn(view, snapshot_day):
    tx = view.table('transactions')
    d = tx['day'].values.astype(float)
    sv = tx['sales_value'].values.astype(float)
    hh = tx['household_key'].values
    bid = tx['basket_id'].values
    out = pd.DataFrame(index=pd.Index(sorted(tx['household_key'].unique()), name='household_key'))
    for hl in (14, 28, 56, 112):
        lam = np.log(2.0)/hl
        w = np.exp(-lam*(snapshot_day - d))
        df = pd.DataFrame({'hh': hh, 'sw': sv*w, 'wl': np.log1p(np.maximum(sv,0))*w, 'w': w})
        agg = df.groupby('hh').agg(ew_spend=('sw','sum'), ew_log=('wl','sum'), wmass=('w','sum'))
        out['ew_spend_hl%d'%hl] = agg['ew_spend']
        out['ew_rate_hl%d'%hl] = agg['ew_spend']*lam
        out['ew_log_hl%d'%hl] = agg['ew_log']/agg['wmass']
        # basket-level weighted count
        b = pd.DataFrame({'hh': hh, 'bid': bid, 'day': d}).drop_duplicates('bid')
        bw = np.exp(-lam*(snapshot_day - b['day'].values))
        out['ew_bask_hl%d'%hl] = pd.Series(bw, index=b['hh']).groupby(level=0).sum()
    return out

cand = agent_api.build_features(fn)
print('cand', cand.shape)

base = agent_api.load_saved('e007_te.parquet')
tt = agent_api.train_targets()
d = base.merge(cand.reset_index(), on=['household_key','snapshot_day']).merge(tt, on=['household_key','snapshot_day'])
print('merged', d.shape)
agent_api.save_table(d.drop(columns=['future_spend_4w']), 'e009_ewma.parquet')
print('saved e009_ewma.parquet')


# ---- cell ----
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
