import agent_api as api
import pandas as pd, numpy as np

tt = api.train_targets()
e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=['household_key','snapshot_day'], how='inner').copy()
y = m['future_spend_4w'].values

# residualize y on a compact set of E015's strongest predictors
X = m[['e13','ew28','usual13_4','mfly','spend_84','b75']].copy()
X = X.fillna(X.median())
Xm = np.c_[np.ones(len(X)), X.values]
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
resid = y - Xm @ beta
print('resid std:', resid.std().round(2), ' y std:', y.std().round(2), ' y mean:', y.mean().round(1))

def resid_corr(cols, df):
    out = []
    for c in cols:
        x = pd.to_numeric(df[c], errors='coerce')
        if x.notna().sum() < 500: 
            out.append((c, np.nan, x.notna().sum())); continue
        out.append((c, np.corrcoef(x.fillna(x.median()), resid)[0,1], x.notna().sum()))
    return pd.DataFrame(out, columns=['feat','resid_corr','n_ok']).sort_values('resid_corr', key=lambda s: s.abs(), ascending=False)

e03 = api.load_saved('e003_dept_mix.parquet')
e02 = api.load_saved('e002_marketing.parquet')
e04 = api.load_saved('e004_long_hist.parquet')
e10 = api.load_saved('e010_composite.parquet')
bf  = api.baseline_features()

key = ['household_key','snapshot_day']
m3 = m[key].copy(); m3['resid'] = resid

for name, tbl in [('e003', e03), ('e002', e02), ('e004', e04), ('e010', e10)]:
    cols = [c for c in tbl.columns if c not in e15.columns and c not in key and c != 'spend_28']
    if not cols: print(name, 'no new cols'); continue
    t = m3.merge(tbl[key+cols], on=key, how='left')
    rc = resid_corr(cols, t)
    print(f'\n== {name}: top 12 by |resid corr| of {len(cols)} new cols')
    print(rc.head(12).round(4).to_string(index=False))

# demographics + calendar as numeric codes
t = m3.merge(bf[key+['classification_2','snapshot_day_index','week_of_year','classification_1','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']], on=key, how='left')
t['c2_num'] = t['classification_2'].map({'X':0,'Y':1,'Z':2})
rc = resid_corr(['c2_num','snapshot_day_index','week_of_year'], t)
print('\n== baseline numeric extras'); print(rc.round(4).to_string(index=False))
for c in ['classification_1','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
    g = t.groupby(c)['resid'].agg(['mean','count'])
    g = g[g['count']>300]
    if len(g)>2: print(c, 'resid mean spread:', (g['mean'].max()-g['mean'].min()).round(1))