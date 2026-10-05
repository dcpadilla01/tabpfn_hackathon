
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
preds = {}
for n in names:
    df = agent_api.load_saved(n + '.parquet')
    preds[n] = df
    print(n, df.shape, list(df.columns))

tt = agent_api.train_targets()
print('train_targets', tt.shape, list(tt.columns))

fv4 = agent_api.load_saved('feats_v4.parquet')
print('feats_v4', fv4.shape, list(fv4.columns)[:40])


# ---- cell ----

import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}

# check alignment
base = P['pred_e013'][['household_key','snapshot_day']].copy()
base = base.sort_values(['household_key','snapshot_day']).reset_index(drop=True)
for n in names:
    q = P[n][['household_key','snapshot_day']].sort_values(['household_key','snapshot_day']).reset_index(drop=True)
    assert q.equals(base), n
print('all aligned, rows:', len(base))

M = pd.DataFrame({n: P[n]['prediction'].values for n in names})
print(M.describe().T[['mean','std','min','max']])

C = M.corr()
print(C.round(4))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}

ref = P['pred_e013'][['household_key','snapshot_day']].copy()
ref['k'] = ref.household_key.astype(str) + '_' + ref.snapshot_day.astype(str)

M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy()
    d['k'] = d.household_key.astype(str) + '_' + d.snapshot_day.astype(str)
    m = ref.merge(d[['k','prediction']], on='k', how='left')
    print(n, 'missing:', m.prediction.isna().sum(), 'dup k:', d.k.duplicated().sum())
    M[n] = m.prediction.values

print(M.describe().T[['mean','std','min','max']].round(2))
C = M.corr()
print(C.round(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
import itertools, math

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}

ref = P['pred_e013'][['household_key','snapshot_day']].copy()
ref['k'] = ref.household_key.astype(str) + '_' + ref.snapshot_day.astype(str)

M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy()
    d['k'] = d.household_key.astype(str) + '_' + d.snapshot_day.astype(str)
    m = ref.merge(d[['k','prediction']], on='k', how='left')
    M[n] = m.prediction.values

tt = agent_api.train_targets()
tt['k'] = tt.household_key.astype(str) + '_' + tt.snapshot_day.astype(str)
y_map = dict(zip(tt.k, tt.future_spend_4w))
y = ref.k.map(y_map).values
print('y NaN:', np.isnan(y).sum(), 'n:', len(y))

# per-model val MAE
for n in names:
    p = M[n].values
    print(n, round(np.mean(np.abs(p - y)), 3))
print('mean y:', y.mean().round(2), 'median y:', np.median(y).round(2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
ref['k'] = ref.household_key.astype(str) + '_' + ref.snapshot_day.astype(str)
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy(); d['k'] = d.household_key.astype(str) + '_' + d.snapshot_day.astype(str)
    M[n] = ref.merge(d[['k','prediction']], on='k', how='left').prediction.values

tt = agent_api.train_targets()
tt['k'] = tt.household_key.astype(str) + '_' + tt.snapshot_day.astype(str)
ref2 = ref[ref.k.isin(set(tt.k))].copy()
Mtr = M.loc[ref2.index]
y = ref2.k.map(dict(zip(tt.k, tt.future_spend_4w))).values
print('train rows:', len(y), 'snapdays:', sorted(ref2.snapshot_day.unique()))

for n in names:
    p = Mtr[n].values
    mae_all = np.mean(np.abs(p-y))
    per = []
    for d in sorted(ref2.snapshot_day.unique()):
        m = ref2.snapshot_day.values==d
        per.append(f"{d}:{np.mean(np.abs(p[m]-y[m])):.0f}")
    print(n, 'MAE', round(mae_all,2), ' '.join(per))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

for f in ['feats_v1','feats_v2','feats_v3','feats_v4']:
    d = agent_api.load_saved(f + '.parquet')
    print(f, d.shape, 'snapdays:', sorted(d.snapshot_day.unique()))
    print('  cols:', list(d.columns))

tt = agent_api.train_targets()
print('tt days:', sorted(tt.snapshot_day.unique()), tt.shape)


# ---- cell ----

import agent_api, pandas as pd, numpy as np

P13 = agent_api.load_saved('pred_e013.parquet')
tt = agent_api.train_targets()
print(P13.dtypes)
print(tt.dtypes)
print(P13.head(3))
print(tt.head(3))
print(P13.household_key.unique()[:5], tt.household_key.unique()[:5])


# ---- cell ----

import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy()
    M[n] = ref.merge(d, on=['household_key','snapshot_day'], how='left').prediction.values

tt = agent_api.train_targets()
tr = tt.merge(ref, on=['household_key','snapshot_day'], how='inner')
print('train rows matched:', len(tr), 'days:', sorted(tr.snapshot_day.unique()))
Mtr = ref.merge(tt, on=['household_key','snapshot_day']).merge(
    pd.DataFrame({n: M[n] for n in names}).assign(household_key=ref.household_key, snapshot_day=ref.snapshot_day),
    on=['household_key','snapshot_day'])
y = Mtr.future_spend_4w.values
print('n=', len(y))
for n in names:
    p = Mtr[n].values
    per = []
    for d in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
        m = Mtr.snapshot_day.values==d
        per.append(f"{d%100}:{np.mean(np.abs(p[m]-y[m])):.0f}")
    print(n, 'MAE', round(np.mean(np.abs(p-y)),2), ' '.join(per))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

names = ['pred_e004','pred_e005','pred_e007','pred_e008','pred_e011',
         'pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n]
    M[n] = ref.merge(d, on=['household_key','snapshot_day'], how='left').prediction.values
print('NaNs per model:', M.isna().sum().values, 'rows:', len(M))

blend = M.mean(axis=1).clip(lower=0).values
print('blend mean/std/min/max:', blend.mean().round(2), blend.std().round(2), blend.min().round(2), blend.max().round(2))

out = ref.copy()
out['prediction'] = blend
p = agent_api.save_table(out, 'pred_e018.parquet')
print('saved:', p)
