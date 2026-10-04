import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e012_robust.parquet')
print(t.shape)
print(t.dtypes.value_counts())
print(t.columns.tolist()[:50])
print('has keys:', 'household_key' in t.columns, 'snapshot_day' in t.columns)
print(t['snapshot_day'].value_counts().sort_index())
print('dup rows:', t.duplicated(['household_key','snapshot_day']).sum())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e012_robust.parquet')
cols = t.columns.tolist()
# print all columns in chunks
for i in range(0, len(cols), 25):
    print(i, cols[i:i+25])


# ---- cell ----
import agent_api, pandas as pd, numpy as np
sd = agent_api.snapshot_days(); print(sd)
tt = agent_api.train_targets()
print('targets:', tt.shape); print(tt.head(3))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','std','count']).round(1))
e009 = agent_api.load_saved('e009_ewma_longlags.parquet')
print('e009:', e009.shape)
print(e009.groupby('snapshot_day').size())
v = agent_api.snapshot(as_of_day=207)
print('households type:', type(v.households))
hh207 = set(np.asarray(v.households).ravel().tolist())
fd = v.transactions.groupby('household_key')['day'].min()
pred = set(fd[fd <= 207-84].index)
print('hh207', len(hh207), 'pred', len(pred), 'only_view', len(hh207-pred), 'only_pred', len(pred-hh207))
sub = e009[e009.snapshot_day==207]
print('tlag NaN rates:', sub[['spend_28','tlag_2','tlag_5','tlag_13']].isna().mean().round(3).to_dict())
print(sub.groupby(sub['tlag_13'].isna())['tenure_days'].agg(['mean','count']))
demo = agent_api.snapshot().demographics
print('demo:', demo.shape); print(demo.head(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
# Probe: are there households with no purchase in the 28d before snapshot? (churn/zero-window)
# and what does the target look like for them vs active ones
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
print('merged:', m.shape)
for c in ['spend_28','spend_56','spend_84','days_since_last','tenure_days']:
    print(c, 'NaN:', m[c].isna().mean().round(4))
# zero-window rate by snapshot day
m['zero28'] = (m['spend_28']==0).astype(int)
print(m.groupby('snapshot_day')['zero28'].mean().round(3))
# target stats by zero28
print(m.groupby('zero28')['future_spend_4w'].agg(['mean','std','count']).round(1))
# corr of spend_28 with target
print('corr spend_28 vs target:', m[['spend_28','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_84 vs target:', m[['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_364 vs target:', m[['spend_364','future_spend_4w']].corr().iloc[0,1].round(3))
# distribution of target
print(m['future_spend_4w'].describe().round(1))
print('target quantiles:', m['future_spend_4w'].quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).round(0).to_dict())
print('zero target share:', (m['future_spend_4w']==0).mean().round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>=459]
print('train rows', len(tr), 'val rows', len(va))
for c in ['spend_84','spend_28','spend_56','ewma_4']:
    print(c, 'val MAE:', round(mae(va['future_spend_4w'], va[c]),3))
z = va[va['spend_28']==0]
print('val zero28 rows:', len(z), 'target mean:', round(z['future_spend_4w'].mean(),1), 'median:', z['future_spend_4w'].median())
print('val overall mean target:', round(va['future_spend_4w'].mean(),1))
pred = np.where(va['spend_28']==0, 0, va['spend_84'])
print('hybrid MAE:', round(mae(va['future_spend_4w'], pred),3))
pred2 = np.where(va['spend_28']==0, 0, tr[tr['spend_28']>0]['future_spend_4w'].mean())
print('hybrid2 MAE:', round(mae(va['future_spend_4w'], pred2),3))
print('corr spend_84-target (zero28 rows):', m[m['spend_28']==0][['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_84-target (active rows):', m[m['spend_28']>0][['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
# zero28 rows: distribution of future target
print('zero28 target quantiles:', m[m['spend_28']==0]['future_spend_4w'].quantile([0.25,0.5,0.75,0.9]).round(0).to_dict())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431].copy()
print('train rows', len(tr))
print('MAE spend_84:', round(mae(tr.future_spend_4w, tr.spend_84),2))
print('MAE spend_28:', round(mae(tr.future_spend_4w, tr.spend_28),2))
print('MAE ewma_4:', round(mae(tr.future_spend_4w, tr.ewma_4),2))
print('MAE mean(train):', round(mae(tr.future_spend_4w, np.full(len(tr), tr.future_spend_4w.mean())),2))
# hybrid: 0 for zero28 else spend_84
pred = np.where(tr.spend_28==0, 0, tr.spend_84)
print('MAE hybrid zero28|spend84:', round(mae(tr.future_spend_4w, pred),2))
# per-snapshot-day mean target (calendar effect)
day_mean = tr.groupby('snapshot_day')['future_spend_4w'].mean()
print(day_mean.round(1).to_dict())
# how much of MAE comes from zero28 rows? if we predict their conditional mean:
zmean = tr[tr.spend_28==0].future_spend_4w.mean()
pred2 = np.where(tr.spend_28==0, zmean, tr.spend_84)
print('MAE hybrid zero28-mean|spend84:', round(mae(tr.future_spend_4w, pred2),2))
# error decomposition proxy: MAE within zero28 rows using best constant
z = tr[tr.spend_28==0]
print('zero28 rows: MAE with constant mean:', round(mae(z.future_spend_4w, np.full(len(z), zmean)),2),
      'MAE with 0:', round(mae(z.future_spend_4w, np.zeros(len(z))),2))
a = tr[tr.spend_28>0]
amean = a.future_spend_4w.mean()
print('active rows: MAE const:', round(mae(a.future_spend_4w, np.full(len(a), amean)),2),
      'MAE spend_84:', round(mae(a.future_spend_4w, a.spend_84),2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]
y = tr.future_spend_4w.values
# candidate single/composite predictors (all same scale)
cands = {
 'spend_28': tr.spend_28.values,
 'tlag_mean': tr.tlag_mean.values,
 'ewma_4': tr.ewma_4.values,
 'blend_s28_tlmean': 0.5*tr.spend_28.values+0.5*tr.tlag_mean.values,
 'blend_s28_tl_ewma': 0.4*tr.spend_28.values+0.3*tr.tlag_mean.values+0.3*tr.ewma_4.values,
 'mean_tlag_nonzero': (tr.tlag_mean.values),
}
for k,v in cands.items():
    print(k, 'MAE:', round(mae(y, v),2))
# tlag window definition check: correlation matrix of tlags with target
tc = ['tlag_2','tlag_3','tlag_4','tlag_5','tlag_6','tlag_7','tlag_8']
print(tr[tc+['spend_28','future_spend_4w']].corr()['future_spend_4w'].round(3))
# among active rows only
a = tr[tr.spend_28>0]
ya = a.future_spend_4w.values
print('--- active rows ---')
for k,v in {'spend_28':a.spend_28.values,'tlag_mean':a.tlag_mean.values,
            'blend':0.5*a.spend_28.values+0.5*a.tlag_mean.values}.items():
    print(k, round(mae(ya,v),2))
# zero28 rows
z = tr[tr.spend_28==0]
print('--- zero28 rows ---')
print('MAE 0:', round(mae(z.future_spend_4w, np.zeros(len(z))),2),
      'MAE tlag_mean:', round(mae(z.future_spend_4w, z.tlag_mean.values),2),
      'MAE 0.3*tlag_mean:', round(mae(z.future_spend_4w, 0.3*z.tlag_mean.values),2))
# share of target==0 among active
print('share target==0 among active:', (a.future_spend_4w==0).mean().round(3))
print('share target==0 among zero28:', (z.future_spend_4w==0).mean().round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags k'.replace(' k','')).copy() if False else agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]
y = tr.future_spend_4w.values
# Try a few ridge-like blends via grid search on train (proxy for val)
best=None
for w1 in [0.3,0.4,0.5,0.6,0.7]:
    for w2 in [0.1,0.2,0.3,0.4]:
        w3 = 1-w1-w2
        if w3<0: continue
        p = w1*tr.spend_28.values+w2*tr.tlag_mean.values+w3*tr.ewma_4.values
        v = mae(y,p)
        if best is None or v<best[0]: best=(v,w1,w2,w3)
print('best blend:', best)
# tlag_2 alone vs spend_28
print('tlag_2 MAE:', round(mae(y, tr.tlag_2.values),2))
# robust: clipped spend_28
for cap in [400,600,800]:
    print(f'clip(spend_28,{cap}) MAE:', round(mae(y, np.minimum(tr.spend_28.values,cap)),2))
# per-day mean target as baseline
day_mean = tr.groupby('snapshot_day')['future_spend_4w'].mean()
p = tr.snapshot_day.map(day_mean).values
print('MAE per-day mean:', round(mae(y,p),2))
# is target autocorrelated with day? check day 431 vs 95
print(tr.groupby('snapshot_day')['future_spend_4w'].mean().diff().round(2).to_dict())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
# Dry-run build_features to see what a view exposes inside fn
def probe(view, s):
    print('snapshot_day', s, 'day', view.day, 'week', view.week)
    print('households type:', type(view.households))
    hh = view.households
    try:
        print('len hh:', len(hh), 'first:', list(hh)[:3])
    except Exception as e:
        print('hh err', e)
    tx = view.transactions
    print('tx shape', tx.shape, 'max day', tx['day'].max(), 'min day', tx['day'].min())
    print('neg sales:', (tx.sales_value<0).sum(), 'zero sales:', (tx.sales_value==0).sum())
    print('cols:', tx.columns.tolist())
    return pd.DataFrame(index=hh if hh is not None else tx.household_key.unique())

out = agent_api.build_features(probe)
print('out shape:', out.shape)
print(out.head())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def probe(view, s):
    print('day', view.day, 'week', view.week)
    tx = view.transactions
    print('tx', tx.shape, 'days', tx.day.min(), tx.day.max())
    hh = np.asarray(view.households).ravel().tolist() if view.households is not None else tx.household_key.unique()
    print('n hh', len(hh), 'type', type(view.households))
    # first purchase day per household
    fd = tx.groupby('household_key')['day'].min()
    print('hh with first purchase <=', s-84, ':', (fd<=s-84).sum(), 'of', len(fd))
    print('example hh days:', fd.head())
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print(out.shape)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
def probe(view, s):
    tx = view.transactions
    hh = np.asarray(view.households).ravel().tolist() if view.households is not None else tx.household_key.unique()
    fd = tx.groupby('household_key')['day'].min()
    n_elig = int((fd<=s-84).sum()); n_hh = len(fd)
    print(f's={s}: households={len(hh)} tx={len(tx)} first<=s-84: {n_elig}/{n_hh}')
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print('done', out.shape)
