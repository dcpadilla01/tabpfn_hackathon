import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
demo = agent_api.snapshot().demographics.copy()
for c in demo.columns:
    if str(demo[c].dtype) == 'category': demo[c] = demo[c].astype(str)

d = demo.copy()
d['age_ord'] = pd.to_numeric(d['classification_1'].str.extract(r"(\d+)")[0], errors='coerce')
d['income_ord'] = pd.to_numeric(d['classification_3'].str.extract(r"(\d+)")[0], errors='coerce')
d['size_ord'] = pd.to_numeric(d['classification_4'].str.extract(r"(\d+)")[0], errors='coerce')
d['grp_ord'] = pd.to_numeric(d['classification_5'].str.extract(r"(\d+)")[0], errors='coerce')
d['kid_ord'] = d['kid_category_desc'].map({'None/Unknown':0,'1':1,'2':2,'3+':3}).astype(float)
oh = pd.get_dummies(d[['classification_2','homeowner_desc','kid_category_desc']].astype(str), prefix=['c2','ho','kid']).astype(float)
demoF = pd.concat([d[['household_key','age_ord','income_ord','size_ord','grp_ord','kid_ord']], oh], axis=1)
demoF['has_demo'] = 1.0
for c in demoF.columns:
    if c!='household_key': demoF[c] = pd.to_numeric(demoF[c], errors='coerce').fillna(0.0)

comb = e008.merge(demoF, on='household_key', how='left')
for c in demoF.columns:
    if c!='household_key': comb[c] = comb[c].fillna(0.0)
L84 = np.log1p(comb['spend_84'].clip(lower=0)); L28 = np.log1p(comb['spend_28'].clip(lower=0))
for c in ['age_ord','income_ord','size_ord','grp_ord','kid_ord']:
    comb['ix_'+c] = comb[c]*L84
comb['ix_size_L28'] = comb['size_ord']*L28
print('comb', comb.shape, 'dtypes ok:', all(np.issubdtype(t, np.number) for t in comb.dtypes if t!=object))

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

c02 = comb.merge(e002.drop(columns=[c for c in e002.columns if c in comb.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('fwd  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('fwd  +demo       v431: %.3f | v403+431: %.3f' % (fwd_mae(comb,[431]), fwd_mae(comb,[403,431])))
print('fwd  +demo+mkt   v431: %.3f | v403+431: %.3f' % (fwd_mae(c02,[431]), fwd_mae(c02,[403,431])))
print('LOO  E008   : %.3f' % loo_mae(e008))
print('LOO  +demo  : %.3f' % loo_mae(comb))
print('LOO  +demo+mkt: %.3f' % loo_mae(c02))
agent_api.save_table(comb, 'e009_demo.parquet')
agent_api.save_table(c02, 'e009_demo_mkt.parquet')
print('saved both')