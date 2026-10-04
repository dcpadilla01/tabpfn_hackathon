import agent_api as api
import pandas as pd, numpy as np

for t in ['e015_base','e013_union','e009_macro','cand_new','e006_seq_gaps']:
    df = api.load_saved(t + '.parquet')
    print('==', t, df.shape)
    print(list(df.columns))
    print()

bf = api.baseline_features()
print('== baseline_features', bf.shape)
print(list(bf.columns))
print(bf.head(3))
print()

for t in ['e001_history','e002_marketing','e003_dept_mix','e004_long_hist','e005_seasonal_peer','e007_new','e008_decomp2','e010_composite','e011_display','e014_base']:
    df = api.load_saved(t + '.parquet')
    print('==', t, df.shape, list(df.columns))
    print()

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

bf = api.baseline_features()
print('baseline cols:', list(bf.columns))

tt = api.train_targets()
print('train_targets:', tt.shape, list(tt.columns))

e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)

num = [c for c in e15.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
corrs = {}
for c in num:
    x = m[c]
    if x.notna().sum() > 100:
        corrs[c] = x.corr(m['future_spend_4w'])
cs = pd.Series(corrs).sort_values()
print('\n--- E015 weakest |corr| (bottom 25):')
print(cs.abs().sort_values().head(25).round(4))
print('\n--- E015 strongest |corr| (top 20):')
print(cs.abs().sort_values().tail(20).round(4))

# missing families: display/mailer (e011), composites (e010)
e11 = api.load_saved('e011_display.parquet')
e10 = api.load_saved('e010_composite.parquet')
extra11 = [c for c in e11.columns if c not in e15.columns and c not in ('household_key','snapshot_day')]
extra10 = [c for c in e10.columns if c not in e15.columns and c not in ('household_key','snapshot_day')]
print('\nextra11:', extra11)
print('extra10:', extra10)
m2 = m.merge(e11[['household_key','snapshot_day']+extra11], on=['household_key','snapshot_day'], how='left')
m2 = m2.merge(e10[['household_key','snapshot_day']+extra10], on=['household_key','snapshot_day'], how='left')
for c in extra11+extra10:
    x = m2[c]
    if x.notna().sum() > 100 and pd.api.types.is_numeric_dtype(x):
        print(f'{c:18s} corr={x.corr(m2["future_spend_4w"]):.4f}  corr_with_e13={x.corr(m2["e13"]):.3f}')

# baseline calendar/demographic corr
mb = m.merge(bf.drop(columns=[c for c in bf.columns if c in e15.columns]), on=['household_key','snapshot_day'], how='left')
for c in ['snapshot_day_index','week_of_year','classification_2']:
    if c in mb.columns:
        x = mb[c]
        print(f'{c:20s} dtype={x.dtype} nunique={x.nunique()} corr={pd.to_numeric(x, errors="coerce").corr(mb["future_spend_4w"]) if pd.api.types.is_numeric_dtype(x) else "cat"}')
print('\ncalls ok')

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

tt = api.train_targets()
e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')

e11 = api.load_saved('e011_display.parquet')
e10 = api.load_saved('e010_composite.parquet')
bf  = api.baseline_features()

extra11 = [c for c in e11.columns if c not in e15.columns and c not in ('household_key','snapshot_day','spend_28')]
extra10 = [c for c in e10.columns if c not in e15.columns and c not in ('household_key','snapshot_day','spend_28')]
print('extra11:', extra11); print('extra10:', extra10)

m2 = m.merge(e11[['household_key','snapshot_day']+extra11], on=['household_key','snapshot_day'], how='left')
m2 = m2.merge(e10[['household_key','snapshot_day']+extra10], on=['household_key','snapshot_day'], how='left')
print('\n-- display/composite feature correlations with target and with e13:')
for c in extra11+extra10:
    x = m2[c]
    if pd.api.types.is_numeric_dtype(x) and x.notna().sum() > 100:
        print(f'{c:16s} corr_y={x.corr(m2["future_spend_4w"]):.4f}  corr_e13={x.corr(m2["e13"]):.3f}  na={x.isna().mean():.2f}')
    else:
        print(f'{c:16s} dtype={x.dtype} nunique={x.nunique()}')

mb = m.merge(bf[['household_key','snapshot_day','classification_2','snapshot_day_index','week_of_year']],
              on=['household_key','snapshot_day'], how='left')
print('\n-- baseline extras:')
print('classification_2 values:', mb['classification_2'].value_counts(dropna=False).to_dict())
for c in ['snapshot_day_index','week_of_year']:
    print(c, 'corr_y=', mb[c].corr(mb['future_spend_4w']).round(4))

# mean target by classification_2
print('\nmean target by classification_2:')
print(mb.groupby('classification_2')['future_spend_4w'].agg(['mean','count']).round(1))

# check 'streak' NaN corr (constant?) and missingness of top features
print('\nstreak nunique:', m['streak'].nunique(), 'na:', m['streak'].isna().mean())
top = ['e13','e6','b75','usual13','ew28','ew56','spend_84','weekly_rate_84','e13_4','usual13_4','mfly','w1','w13']
print('\nmissingness of top features:')
print(m[top].isna().mean().round(3))
print('\ndone')

# ---- cell ----
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

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

key = ['household_key','snapshot_day']
tt = api.train_targets()
e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=key, how='inner').copy()
y = m['future_spend_4w'].values

# residualize target on E015's strongest predictors
X = m[['e13','ew28','usual13_4','mfly','spend_84','b75']].copy().fillna(0)
Xm = np.c_[np.ones(len(X)), X.values]
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
resid = y - Xm @ beta

# candidate pool of NEW features (not in E015)
e10 = api.load_saved('e010_composite.parquet').set_index(key)
e01 = api.load_saved('e001_history.parquet').set_index(key)
e04 = api.load_saved('e004_long_hist.parquet').set_index(key)
e03 = api.load_saved('e003_dept_mix.parquet').set_index(key)
bf  = api.baseline_features().set_index(key)

pool = {
 'act_persist': e10['act_persist'], 'us_stab': e10['us_stab'], 'log_act_usual': e10['log_act_usual'],
 'spend_7': e01['spend_7'], 'spend_28': e01['spend_28'], 'log_spend_28': e01['log_spend_28'],
 'days_since_last': e01['days_since_last'], 'qty_28': e01['qty_28'], 'spend_prev28': e01['spend_prev28'],
 'trend28': e01['trend28'], 'spend_all': e01['spend_all'], 'spend_ly28': e01['spend_ly28'],
 'nb_56': e04['nb_56'], 'std_wk8': e04['std_wk8'], 'max_wk8': e04['max_wk8'], 'wk_rate_all': e04['wk_rate_all'],
 'd84_SALAD BAR': e03['d84_SALAD BAR'], 'd84_GARDEN CENTER': e03['d84_GARDEN CENTER'],
 'd84_DELI': e03['d84_DELI'], 'd84_HBC': e03['d84_HBC'], 'd84_MEAT-PCKGD': e03['d84_MEAT-PCKGD'],
 'dept_entropy_84': e03['dept_entropy_84'], 'zero_week_share_84': e03['zero_week_share_84'],
 'spend_vol_84': e03['spend_vol_84'],
 'classification_2': bf['classification_2'], 'snapshot_day_index': bf['snapshot_day_index'],
 'week_of_year': bf['week_of_year'],
}
pool = pd.DataFrame(pool)

# residual correlations on train rows
idx = m.set_index(key).index
p_tr = pool.loc[idx]
cors = {}
for c in pool.columns:
    x = pd.to_numeric(p_tr[c], errors='coerce')
    if x.notna().sum() < 500:
        cors[c] = np.nan
    else:
        cors[c] = np.corrcoef(x.fillna(x.median()).values, resid)[0,1]
cs = pd.Series(cors)
print('residual corrs:'); print(cs.sort_values(key=lambda s: s.abs(), ascending=False).round(4))

# select: |resid corr| >= 0.018, plus always calendar + classification_2 + spend_28
sel = [c for c in cs.index if (abs(cs[c]) >= 0.018)]
for c in ['classification_2','snapshot_day_index','week_of_year','spend_28']:
    if c not in sel: sel.append(c)
print('\nselected:', sel, len(sel))

# build final table: E015 base (drop constant 'streak') + selected
base = e15.drop(columns=['streak'])
add = pool[sel].reset_index()
final = base.merge(add, on=key, how='left')
print('final shape:', final.shape, ' dup rows:', final.duplicated(key).sum())
print('na frac of new cols (max):', final[sel].isna().mean().max().round(3))
path = api.save_table(final, 'e017_gapfill')
print('saved:', path)