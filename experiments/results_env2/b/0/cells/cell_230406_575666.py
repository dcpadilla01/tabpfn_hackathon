
import numpy as np, pandas as pd, collections

api = agent_api
df = api.load_saved('e011_pruned_basket.parquet')
assert df is not None, 'load failed'
tt = api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
y_all = m['future_spend_4w'].astype(float).values
print('rows', len(m), 'feats', len(feats), 'zero_share', round(float((y_all==0).mean()),4))
print(m['future_spend_4w'].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_string())

# encode features (aligned to m rows)
cols = []
for c in feats:
    s = m[c]
    if pd.api.types.is_numeric_dtype(s):
        cols.append(pd.to_numeric(s, errors='coerce'))
    else:
        cols.append(pd.Series(pd.factorize(s)[0], index=m.index).replace(-1, np.nan))
Xdf = pd.concat(cols, axis=1); Xdf.columns = feats
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)

prio = [
 ('calendar', ['snapshot_day','week']),
 ('demo', ['classification','homeowner','kid_category','has_demographics']),
 ('marketing', ['campaign','coupon']),
 ('gapzero', ['gap','zero','active_']),
 ('trend', ['decay','ewm','hl','ratio','last_year','weekly','slope','mom','trend']),
 ('mix', ['dept','brand','unit_price','distinct','n_products','variety','top']),
 ('history', ['spend','trip','days_since','tenure','lines','basket','store','max_']),
]
def grp(c):
    for g, subs in prio:
        if any(s in c for s in subs):
            return g
    return 'other'
gmap = {c: grp(c) for c in feats}
print(collections.Counter(gmap.values()))
for g in sorted(set(gmap.values())):
    print('##', g, ':', sorted([c for c in feats if gmap[c]==g]))

days_train = api.snapshot_days()['train']
H1 = [403, 431]; T1 = [d for d in days_train if d not in H1]
H2 = [347, 375, 403, 431]; T2 = [d for d in days_train if d not in H2]

def run(feat_list, train_days, hold_days, lam):
    tr = m['snapshot_day'].isin(train_days).values
    va = m['snapshot_day'].isin(hold_days).values
    Xtr = Xdf.loc[tr, feat_list].values.astype(float)
    Xva = Xdf.loc[va, feat_list].values.astype(float)
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    ytr = y_all[tr]; yva = y_all[va]
    Xb = np.hstack([Xtr, np.ones((len(Xtr),1))])
    p = Xb.shape[1]; pen = np.eye(p)*lam; pen[-1,-1]=0
    w = np.linalg.solve(Xb.T@Xb + pen, Xb.T@ytr)
    pred = np.hstack([Xva, np.ones((len(Xva),1))]) @ w
    return float(np.abs(pred-yva).mean())

lams = [10,30,100,300,1000,3000]
print('--- lambda grid (all feats, H1=[403,431]) ---')
scores = {lam: run(feats, T1, H1, lam) for lam in lams}
print({k: round(v,3) for k,v in scores.items()})
best_lam = min(scores, key=scores.get)
full = scores[best_lam]
print('best_lam', best_lam, 'full H1', round(full,3), '| full H2', round(run(feats, T2, H2, best_lam),3))

print('--- group ablation (drop group, H1) delta = MAE_drop - full ---')
for g in sorted(set(gmap.values())):
    sub = [c for c in feats if gmap[c]!=g]
    d = run(sub, T1, H1, best_lam)
    print('drop', g, round(d,3), 'delta', round(d-full,3))

# spearman correlations
sp = []
for c in feats:
    x = Xdf[c].values.astype(float)
    ok = ~np.isnan(x)
    if ok.sum()>10 and np.std(x[ok])>0:
        r = np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y_all[ok]).rank())[0,1]
    else:
        r = np.nan
    sp.append((c, r))
sp_df = pd.DataFrame(sp, columns=['feat','spearman'])
sp_df['grp'] = sp_df.feat.map(gmap)
sp_df = sp_df.sort_values('spearman')
print('--- weakest |rho| ---'); print(sp_df.head(12).to_string())
print('--- strongest ---'); print(sp_df.tail(12).to_string())
nz = sp_df[sp_df.spearman.abs()<0.015].feat.tolist()
print('n near-zero-corr feats:', len(nz))
d = run([c for c in feats if c not in nz], T1, H1, best_lam)
print('drop near-zero-corr set: MAE', round(d,3), 'delta', round(d-full,3))
