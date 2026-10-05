
import agent_api, pandas as pd, numpy as np
e12 = agent_api.load_saved('e012_style.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
rr  = agent_api.load_saved('rawrec.parquet')
print('e12', e12.shape, 'e15', e15.shape, 'rr', rr.shape)
print('e12 per snapshot:\n', e12.groupby('snapshot_day').size())
print('e15 per snapshot:\n', e15.groupby('snapshot_day').size())
print('rr per snapshot:\n', rr.groupby('snapshot_day').size())
print('e12 cols sample:', list(e12.columns)[:12])
print('e15 extra cols:', [c for c in e15.columns if c not in e12.columns])
print('rr cols:', list(rr.columns)[:40])


# ---- cell ----

import agent_api, pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print(rr.dtypes)
print(rr.head(10))
print('nunique hh', rr['household_key'].nunique() if 'household_key' in rr.columns else 'no hh col')


# ---- cell ----

import agent_api, pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print('unique g:', sorted(rr['g'].unique()))
print('g+11 matches snapshot days?', set(np.array(sorted(rr['g'].unique()))+11) == set(agent_api.snapshot_days()['train']+agent_api.snapshot_days()['validation']))
# check for truncation artifact at late g: sp28 zeros share by g
print(rr.groupby('g')['sp28'].apply(lambda s: (s==0).mean()))
print(rr.groupby('g')['sp56'].mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e12 = agent_api.load_saved('e012_style.parquet')
e16 = agent_api.load_saved('e016_merged.parquet')
feat12 = [c for c in e12.columns if c not in ('household_key','snapshot_day')]
allnan = e12[feat12].isna().all(axis=1)
print('e12 rows with ALL features NaN:', allnan.sum())
print(e12.loc[allnan].groupby('snapshot_day').size().head(20))
# compare e16 rawrec cols vs e12 on overlapping non-nan rows
rr = agent_api.load_saved('rawrec.parquet')
common = [c for c in rr.columns if c in e16.columns and c not in ('household_key','g')]
print('n common rawrec cols in e16:', len(common))
sub = e16.merge(rr.rename(columns={'g':'snapshot_day'}), on=['household_key','snapshot_day'], suffixes=('_x','_r'))
print('merged rows', len(sub), 'of', len(e16))
for c in ['sp28','sp56','ew'][:3]:
    if c+'_x' in sub.columns and c+'_r' in sub.columns:
        d = (sub[c+'_x']-sub[c+'_r']).abs()
        print(c, 'median abs diff e12vsrawrec:', d.median(), 'max', d.max())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
feat = [c for c in e16.columns if c not in ('household_key','snapshot_day')]
allnan = e16[feat].isna().all(axis=1)
print('e16 rows with ALL features NaN:', allnan.sum())
print(e16.loc[allnan].groupby('snapshot_day').size())
# count fully-nan per snapshot
cnt = e16.groupby('snapshot_day').apply(lambda d: d[feat].isna().all(axis=1).sum())
print(cnt)


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
print('e16 dup (hh,day) pairs:', e16.duplicated(['household_key','snapshot_day']).sum())
rr = agent_api.load_saved('rawrec.parquet')
print('rawrec dup (hh,g):', rr.duplicated(['household_key','g']).sum())
print('e16 shape', e16.shape, 'unique pairs', len(e16.drop_duplicates(['household_key','snapshot_day'])))
# NaN fraction per feature group
feat = [c for c in e16.columns if c not in ('household_key','snapshot_day')]
nanfrac = e16[feat].isna().mean().sort_values(ascending=False)
print(nanfrac.head(15))
print('features with >50% NaN:', (nanfrac>0.5).sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in ('household_key','snapshot_day','stack_ridge_log','stack_ridge_lin')]
m = e16.merge(e15[k+e12c+['stack_ridge_log','stack_ridge_lin']], on=k, suffixes=('_x','_y'))
print('merged', m.shape)
diffs = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if np.issubdtype(a.dtype, np.number):
        d = (a-b).abs()
        diffs[c] = d.max()
bad = {c:v for c,v in diffs.items() if v > 1e-6}
print('cols differing:', len(bad))
print(list(bad.items())[:20])


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
print([c for c in e15.columns if 'stack' in c])
print([c for c in e16.columns if 'stack' in c])
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in k+['stack_ridge_log','stack_ridge_lin']]
m = e16.merge(e15[k+e12c+['stack_ridge_log']], on=k, suffixes=('_x','_y'))
print('merged', m.shape)
bad = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if np.issubdtype(a.dtype, np.number):
        d = (a-b).abs().max()
        if d > 1e-6: bad[c]=d
print('cols differing:', len(bad), list(bad.items())[:20])
# check stack col present in e16
print('stack in e16:', [c for c in e16.columns if 'stack' in c])


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in k+['stack_ridge_log','stack_ridge']]
m = e16.merge(e15[k+e12c+['stack_ridge_log']], on=k, suffixes=('_x','_y'))
bad = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if pd.api.types.is_numeric_dtype(a):
        d = (a-b).abs().max()
        if d > 1e-6: bad[c]=d
print('numeric cols differing:', len(bad), list(bad.items())[:20])
# categorical compare
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if not pd.api.types.is_numeric_dtype(a):
        neq = (a.astype(str)!=b.astype(str)).sum()
        if neq>0: print('cat diff', c, neq)
print('done')


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
print('e16 n feat:', len(feat), 'max allowed 500 -> ok')
# NaN fraction of the 58 rawrec-only cols
rr = agent_api.load_saved('rawrec.parquet')
rronly = [c for c in rr.columns if c not in k+['g']]
nanfrac = e16[rronly].isna().mean().sort_values(ascending=False)
print('rawrec-only cols:', len(rronly))
print(nanfrac.head(10))
print('rawrec cols with any NaN in e16:', (nanfrac>0).sum())
# where do NaNs come from? rows in e16 not in rawrec merge
m = e16.merge(rr.rename(columns={'g':'snapshot_day'}), on=k, how='left', indicator=True)
print('e16 rows missing from rawrec:', (m['_merge']=='left_only').sum())
print(m.loc[m['_merge']=='left_only'].groupby('snapshot_day').size())


# ---- cell ----

import agent_api, pandas as pd, numpy as all_nan_placeholder


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e15 = agent_api.load_saved('e015_stack.parquet')
print('e15 stack cols:', [c for c in e15.columns if 'stack' in c])
print(e15[['household_key','snapshot_day','stack_ridge_log']].head())
print(e15['stack_ridge_log'].describe())
print('NaN in stack:', e15['stack_ridge_log'].isna().sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
print('e16 n feat:', len(feat))
# NaN fraction per feature for validation-only rows vs train rows
isval = e16['snapshot_day'].isin([459,487,515,543])
nanval = e16.loc[isval, feat].isna().mean()
nantr  = e16.loc[~isval, feat].isna().mean()
d = (nanval-nantr).sort_values(ascending=False)
print('cols with much higher NaN on validation:')
print(d.head(12))
# any feature 100% NaN on validation?
print('100% NaN on validation:', (nanval==1).sum())
print(list(nanval[nanval==1].index)[:20])


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
isval = e16['snapshot_day'].isin([459,487,515,543])
# how many validation rows have ANY NaN among the rawrec cols?
rrcols = [c for c in e16.columns if c in agent_api.load_saved('rawrec.parquet').columns and c not in k]
any_nan = e16.loc[isval, rrcols].isna().any(axis=1)
print('validation rows with any NaN in rawrec cols:', any_nan.sum(), 'of', isval.sum())
print(e16.loc[isval].groupby('snapshot_day').size())
print('frac', any_nan.mean())
# train rows with any NaN
antr = e16.loc[~isval, rrcols].isna().any(axis=1)
print('train rows with any NaN in rawrec cols:', antr.sum(), 'of', (~isval).sum())
