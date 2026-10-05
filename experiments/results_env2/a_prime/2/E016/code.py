import pandas as pd, numpy as np, agent_api

t = agent_api.load_saved('e015_best_pseudo.parquet')
print('shape', t.shape)
print('cols:', list(t.columns))
print(t.dtypes.value_counts())

tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
print('merged', m.shape)
y = m['future_spend_4w']
print('zero frac %.3f mean %.1f' % (y.eq(0).mean(), y.mean()))
print('target quantiles:', y.quantile([.5,.75,.9,.95,.99]).round(1).values)

num = m.drop(columns=['future_spend_4w']).select_dtypes('number')
cor = num.corrwith(y).sort_values()
print('\ncorr with target (sorted):')
print(cor.round(3).to_string())
print('\nn features |corr|<0.01:', (cor.abs()<0.01).sum(), ' |corr|<0.02:', (cor.abs()<0.02).sum())


# ---- cell ----
import pandas as pd, numpy as np, agent_api

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
VAL_DAYS = [459,487,515,543]

def prep(df, tt=None):
    tt = agent_api.train_targets() if tt is None else tt
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    # numeric + one-hot for object/bool
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True)
    X = X.astype(float)
    return X.values, y, m['snapshot_day'].values

def ridge_cv(path_or_df, val_days=(403,431), alphas=(30.,100.,300.,1000.)):
    df = agent_api.load_saved(path_or_df) if isinstance(path_or_df,str) else path_or_df
    X, y, sd = prep(df)
    maes = {}
    for a in alphas:
        errs = []
        for vd in val_days:
            tr = sd < vd; va = sd == vd
            mu, sg = X[tr].mean(0), X[tr].std(0)+1e-9
            Z, Zv = (X[tr]-mu)/sg, (X[va]-mu)/sg
            w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
            p = Zv@w + y[tr].mean()
            errs.append(np.abs(p - y[va]).mean())
        maes[a] = np.mean(errs)
    best = min(maes, key=maes.get)
    return best, maes

for name, path in [('E000','BASELINE'), ('E001','e001_recent_behavior.parquet'),
                   ('E011','e011_discounts.parquet'), ('E012','e012_hh_target_enc.parquet'),
                   ('E015','e015_best_pseudo.parquet')]:
    df = agent_api.baseline_features() if path=='BASELINE' else path
    b, maes = ridge_cv(df)
    print(f'{name}: best_alpha={b:.0f} cvMAE={maes[b]:.3f}  all={ {k:round(v,2) for k,v in maes.items()} }')


# ---- cell ----
import pandas as pd, numpy as np, agent_api

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def prep(df):
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    return X.values, y, m['snapshot_day'].values

def ridge_cv(path_or_df, val_days=(403,431), alphas=(30.,100.,300.,1000.)):
    df = agent_api.load_saved(path_or_df) if isinstance(path_or_df,str) else path_or_df
    X, y, sd = prep(df)
    maes = {}
    for a in alphas:
        errs = []
        for vd in val_days:
            tr = sd < vd; va = sd == vd
            mu = np.nanmean(X[tr],0); sg = np.nanstd(X[tr],0)+1e-9
            Z = np.where(np.isnan(X[tr]), mu, X[tr]); Z = (Z-mu)/sg
            Zv = np.where(np.isnan(X[va]), mu, X[va]); Zv = (Zv-mu)/sg
            w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
            p = Zv@w + y[tr].mean()
            errs.append(np.abs(p - y[va]).mean())
        maes[a] = np.mean(errs)
    b = min(maes, key=maes.get)
    return b, maes

for name, path in [('E000','BASELINE'), ('E001','e001_recent_behavior.parquet'),
                   ('E011','e011_discounts.parquet'), ('E012','e012_hh_target_enc.parquet'),
                   ('E015','e015_best_pseudo.parquet')]:
    df = agent_api.baseline_features() if path=='BASELINE' else path
    b, maes = ridge_cv(df)
    print(f'{name}: alpha={b:.0f} cvMAE={maes[b]:.3f}  all={ {k:round(v,2) for k,v in maes.items()} }')


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def prep(df):
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    return X, y, m['snapshot_day'].values

for name, path in [('E011','e011_discounts.parquet'), ('E015','e015_best_pseudo.parquet')]:
    X, y, sd = prep(agent_api.load_saved(path))
    print(name, X.shape)
    # all-NaN columns overall
    allnan = X.columns[X.isna().all()].tolist()
    print('  all-NaN cols:', allnan)
    # all-NaN within any train fold
    for vd in (403,431):
        tr = sd < vd
        bad = [c for c in X.columns if X.loc[tr, c].isna().all()]
        if bad: print(f'  fold<{vd} all-NaN cols:', bad)
    # check inf
    print('  any inf:', np.isinf(X.values).sum())


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def ridge_cv(path_or_df, val_days=(403,431), alphas=(30.,100.,300.,1000.)):
    df = agent_api.load_saved(path_or_df) if isinstance(path_or_df,str) else path_or_df
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    sd = m['snapshot_day'].values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).values
    maes = {}
    for a in alphas:
        errs = []
        for vd in val_days:
            tr = sd < vd; va = sd == vd
            mu = np.nanmean(X[tr],0); sg = np.nanstd(X[tr],0)+1e-9
            mu = np.where(np.isnan(mu), 0., mu); sg = np.where(np.isnan(sg), 1., sg)
            Z = np.where(np.isnan(X[tr]), mu, X[tr]); Z = (Z-mu)/sg
            Zv = np.where(np.isnan(X[va]), mu, X[va]); Zv = (Zv-mu)/sg
            w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
            p = Zv@w + y[tr].mean()
            errs.append(np.abs(p - y[va]).mean())
        maes[a] = np.mean(errs)
    b = min(maes, key=maes.get)
    return b, maes

for name, path in [('E000','BASELINE'), ('E001','e001_recent_behavior.parquet'),
                   ('E004','e004_temporal.parquet'), ('E005','e005_decay_gapcv.parquet'),
                   ('E010','e010_lifecycle.parquet'), ('E011','e011_discounts.parquet'),
                   ('E012','e012_hh_target_enc.parquet'), ('E014','e014_transforms.parquet'),
                   ('E015','e015_best_pseudo.parquet')]:
    df = agent_api.baseline_features() if path=='BASELINE' else path
    b, maes = ridge_cv(df)
    print(f'{name}: alpha={b:.0f} cvMAE={maes[b]:.3f}  all={ {k:round(v,2) for k,v in maes.items()} }')


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def prep(df):
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    sd = m['snapshot_day'].values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).values
    return X, y, sd

def fit_eval(X, y, sd, vd, a, clip0):
    tr = sd < vd; va = sd == vd
    mu = np.nanmean(X[tr],0); sg = np.nanstd(X[tr],0)+1e-9
    mu = np.where(np.isnan(mu),0.,mu); sg = np.where(np.isnan(sg),1.,sg)
    Z = np.where(np.isnan(X[tr]),mu,X[tr]); Z=(Z-mu)/sg
    Zv = np.where(np.isnan(X[va]),mu,X[va]); Zv=(Zv-mu)/sg
    if a == 0:
        w = np.linalg.lstsq(Z, y[tr]-y[tr].mean(), rcond=None)[0]
    else:
        w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
    p = Zv@w + y[tr].mean()
    if clip0: p = np.clip(p, 0, None)
    return np.abs(p - y[va]).mean()

def cv(path, a, clip0, vds=(403,431)):
    df = agent_api.baseline_features() if path=='BASELINE' else agent_api.load_saved(path)
    X,y,sd = prep(df)
    return np.mean([fit_eval(X,y,sd,vd,a,clip0) for vd in vds])

paths = [('E000','BASELINE'),('E001','e001_recent_behavior.parquet'),('E005','e005_decay_gapcv.parquet'),
         ('E010','e010_lifecycle.parquet'),('E011','e011_discounts.parquet'),
         ('E012','e012_hh_target_enc.parquet'),('E015','e015_best_pseudo.parquet')]
harness = {'E000':92.45,'E001':63.57,'E005':63.32,'E010':62.70,'E011':62.65,'E012':72.18,'E015':62.65}

for a, clip0 in [(0,False),(1,False),(10,False),(1,True),(10,True)]:
    res = {n: cv(p,a,clip0) for n,p in paths}
    ns = list(res); 
    r1 = np.argsort(np.argsort([res[n] for n in ns])); r2 = np.argsort(np.argsort([harness[n] for n in ns]))
    corr = np.corrcoef(r1, r2)[0,1]
    print(f'a={a} clip0={clip0}: ' + ' '.join(f'{n}={res[n]:.2f}' for n in ns) + f'  rankcorr={corr:.2f}')


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def fn(view, s):
    hh = np.asarray(view.households)
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    out = pd.DataFrame(index=pd.Index(hh, name='household_key'))
    g = tx.groupby('household_key')
    last = g['day'].max().reindex(hh)
    out['dsl'] = (s - last).astype(float).values
    for name,(lo,hi) in {'w2':(s-13,s-7),'w3':(s-20,s-14),'w4':(s-27,s-21)}.items():
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        out[name] = v.reindex(hh).fillna(0.0).values
    sp = {}
    for k in range(3):
        lo, hi = s-27-28*k, s-28*k
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        sp[k] = v.reindex(hh).fillna(0.0).values
    out['zeros_3win'] = ((sp[0]==0).astype(float)+(sp[1]==0).astype(float)+(sp[2]==0).astype(float)).values
    s28 = sp[0]
    trips = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key')['day'].diff()
    def gapstats(daymin):
        t = trips[trips.day>=daymin]
        return (t.groupby('household_key')['gap'].median(),
                t.groupby('household_key')['gap'].mean(),
                t.groupby('household_key')['gap'].count())
    med364, mean364, n364 = gapstats(s-363)
    medAll, meanAll, nAll = gapstats(-1)
    med364 = med364.where(n364>=2, medAll); mean364 = mean364.where(n364>=2, meanAll)
    out['med_gap'] = med364.reindex(hh).values
    out['mean_gap'] = mean364.reindex(hh).values
    dsl = out['dsl'].values; mg = out['med_gap'].values; mgA = out['mean_gap'].values
    out['dsl_over_medgap'] = np.where(np.isfinite(mg)&(mg>0), dsl/np.maximum(mg,1e-9), np.nan)
    out['dsl_over_meangap'] = np.where(np.isfinite(mgA)&(mgA>0), dsl/np.maximum(mgA,1e-9), np.nan)
    out['dsl_gt_medgap'] = (dsl > mg)
    w = tx[tx.day>=s-363]
    u = w.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0.0).values
    sv = w.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
    out['units_364'] = u
    out['avg_unit_price_364'] = np.where(u>0, sv/np.maximum(u,1e-9), np.nan)
    out['s28_x_active21'] = s28 * (dsl<=21)
    ang = 2*np.pi*(s+14)/364.0
    out['ann_sin_fw'] = np.sin(ang); out['ann_cos_fw'] = np.cos(ang)
    return out

feat = agent_api.build_features(fn)
print('feat rows', feat.shape)
base = agent_api.load_saved('e015_best_pseudo.parquet').drop(columns=['s28_x_churnrisk'])
merged = base.merge(feat, on=['household_key','snapshot_day'], how='inner')
print('merged', merged.shape, 'nan cols:', merged.isna().all().sum())
tt = agent_api.train_targets()
m = merged.merge(tt, on=['household_key','snapshot_day'])
newc = ['dsl','w2','w3','w4','zeros_3win','med_gap','mean_gap','dsl_over_medgap','dsl_over_meangap','dsl_gt_medgap','units_364','avg_unit_price_364','s28_x_active21','ann_sin_fw','ann_cos_fw']
print(m[newc].corrwith(m['future_spend_4w']).round(3).to_string())
path = agent_api.save_table(merged, 'e016_churn_gapratio.parquet')
print(path)


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def fn(view, s):
    hh = np.asarray(view.households)
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    out = pd.DataFrame(index=pd.Index(hh, name='household_key'))
    g = tx.groupby('household_key')
    last = g['day'].max().reindex(hh)
    out['dsl'] = (s - last).astype(float).values
    for name,(lo,hi) in {'w2':(s-13,s-7),'w3':(s-20,s-14),'w4':(s-27,s-21)}.items():
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        out[name] = v.reindex(hh).fillna(0.0).values
    sp = {}
    for k in range(3):
        lo, hi = s-27-28*k, s-28*k
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        sp[k] = v.reindex(hh).fillna(0.0).values
    out['zeros_3win'] = ((sp[0]==0).astype(float)+(sp[1]==0).astype(float)+(sp[2]==0).astype(float))
    s28 = sp[0]
    trips = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key')['day'].diff()
    def gapstats(daymin):
        t = trips[trips.day>=daymin]
        return (t.groupby('household_key')['gap'].median(),
                t.groupby('household_key')['gap'].mean(),
                t.groupby('household_key')['gap'].count())
    med364, mean364, n364 = gapstats(s-363)
    medAll, meanAll, nAll = gapstats(-1)
    med364 = med364.where(n364>=2, medAll); mean364 = mean364.where(n364>=2, meanAll)
    out['med_gap'] = med364.reindex(hh).values
    out['mean_gap'] = mean364.reindex(hh).values
    dsl = out['dsl'].values; mg = out['med_gap'].values; mgA = out['mean_gap'].values
    out['dsl_over_medgap'] = np.where(np.isfinite(mg)&(mg>0), dsl/np.maximum(mg,1e-9), np.nan)
    out['dsl_over_meangap'] = np.where(np.isfinite(mgA)&(mgA>0), dsl/np.maximum(mgA,1e-9), np.nan)
    out['dsl_gt_medgap'] = (dsl > mg)
    w = tx[tx.day>=s-363]
    u = w.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0.0).values
    sv = w.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
    out['units_364'] = u
    out['avg_unit_price_364'] = np.where(u>0, sv/np.maximum(u,1e-9), np.nan)
    out['s28_x_active21'] = s28 * (dsl<=21)
    ang = 2*np.pi*(s+14)/364.0
    out['ann_sin_fw'] = np.sin(ang); out['ann_cos_fw'] = np.cos(ang)
    return out

feat = agent_api.build_features(fn)
print('feat rows', feat.shape)
base = agent_api.load_saved('e015_best_pseudo.parquet').drop(columns=['s28_x_churnrisk'])
merged = base.merge(feat, on=['household_key','snapshot_day'], how='inner')
print('merged', merged.shape, 'all-nan cols:', merged.isna().all().sum())
tt = agent_api.train_targets()
m = merged.merge(tt, on=['household_key','snapshot_day'])
newc = ['dsl','w2','w3','w4','zeros_3win','med_gap','mean_gap','dsl_over_medgap','dsl_over_meangap','dsl_gt_medgap','units_364','avg_unit_price_364','s28_x_active21','ann_sin_fw','ann_cos_fw']
print(m[newc].corrwith(m['future_spend_4w']).round(3).to_string())
path = agent_api.save_table(merged, 'e016_churn_gapratio.parquet')
print(path)
