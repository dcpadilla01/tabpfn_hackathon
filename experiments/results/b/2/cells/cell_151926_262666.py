import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
snap = agent_api.snapshot()  # capped at 459, fine for static demographics
demo = snap.demographics.copy()
print('demo rows', len(demo), demo.columns.tolist())

d = demo.copy()
d['age_ord'] = d['classification_1'].str.extract(r'(\d+)').astype(float)
d['income_ord'] = d['classification_3'].str.extract(r'(\d+)').astype(float)
d['size_ord'] = d['classification_4'].str.extract(r'(\d+)').astype(float)
d['grp_ord'] = d['classification_5'].str.extract(r'(\d+)').astype(float)
d['kid_ord'] = d['kid_category_desc'].map({'None/Unknown':0,'1':1,'2':2,'3+':3})
for c in ['classification_2','homeowner_desc','kid_category_desc']:
    oh = pd.get_dummies(d[c], prefix=c[:6]).astype(float)
    d = pd.concat([d, oh], axis=1)
demoF = d[['household_key','age_ord','income_ord','size_ord','grp_ord','kid_ord'] +
          [c for c in d.columns if c.startswith('classi_') or c.startswith('homeow') or c.startswith('kid_cat')]].copy()
demoF['has_demo'] = 1.0
print('demoF', demoF.shape)

comb = e008.merge(demoF, on='household_key', how='left')
comb['has_demo'] = comb['has_demo'].fillna(0.0)
for c in ['age_ord','income_ord','size_ord','grp_ord','kid_ord']:
    comb[c] = comb[c].fillna(0.0)
for c in demoF.columns:
    if c not in ('household_key','has_demo') and comb[c].isna().any():
        comb[c] = comb[c].fillna(0.0)
# interactions with log levels
L84 = np.log1p(comb['spend_84'].clip(lower=0)); L28 = np.log1p(comb['spend_28'].clip(lower=0))
for c in ['age_ord','income_ord','size_ord','grp_ord','kid_ord']:
    comb['ix_'+c] = comb[c]*L84
comb['ix_size_L28'] = comb['size_ord']*L28
print('comb', comb.shape)

def fwd_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [dd for dd in tr_days if dd not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(B@w - yva).mean()

def loo_mae(table, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    out=[]
    for dd in tr_days:
        tr = m[m.snapshot_day != dd]; va = m[m.snapshot_day == dd]
        Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
        ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
        mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
        w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        out.append(np.abs(B@w - yva).mean())
    return np.mean(out)

print('fwd  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('fwd  +demo       v431: %.3f' % fwd_mae(comb,[431]))
print('fwd  +demo       v403+431: %.3f  (E008: %.3f)' % (fwd_mae(comb,[403,431]), fwd_mae(e008,[403,431])))
print('LOO  E008   : %.3f' % loo_mae(e008))
print('LOO  +demo  : %.3f' % loo_mae(comb))
path = agent_api.save_table(comb, 'e009_demo.parquet'); print('saved', path)