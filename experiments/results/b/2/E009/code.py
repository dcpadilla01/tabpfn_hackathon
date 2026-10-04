import pandas as pd, numpy as np

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
print('e008 shape', t.shape)
print('dtype counts', t.dtypes.astype(str).value_counts().to_dict())
print('columns:'); print(list(t.columns))

y = tt['future_spend_4w']
print('\ntarget describe:')
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('zero frac', round(float((y==0).mean()),3))

m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('\nmerged', m.shape, 'nan cells', int(m.isna().sum().sum()))

feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(m[c]) and m[c].dtype != bool]
bool_cols = [c for c in feat_cols if m[c].dtype == bool]
cat_cols = [c for c in feat_cols if c not in num_cols and c not in bool_cols]
print('num', len(num_cols), 'bool', len(bool_cols), 'cat', cat_cols)

corr = m[num_cols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print('\ntop |corr| with target:')
print(corr.head(30).round(3))

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
days = agent_api.snapshot_days()
tr_days, va_days = days['train'], days['validation']
print('train days', tr_days); print('val days', va_days)

df = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr = df[df.snapshot_day.isin(tr_days)].reset_index(drop=True)
va = df[df.snapshot_day.isin(va_days)].reset_index(drop=True)
print('train rows', len(tr), 'val rows', len(va))

ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values

def make_X(d, impute):
    X = d[feats].astype(float).copy()
    if impute == 'zero':
        X = X.fillna(0.0)
    else:  # train medians
        med = tr[feats].median()
        X = X.fillna(med)
    return X.values

Xtr_raw = make_X(tr,'zero'); Xva_raw = make_X(va,'zero')
Xtr_med = make_X(tr,'med'); Xva_med = make_X(va,'med')

def fit_eval(Xtr, ytr, Xva, yva, alpha, standardize):
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        Xtr2=(Xtr-mu)/sd; Xva2=(Xva-mu)/sd
    else:
        Xtr2, Xva2 = Xtr, Xva
    Xtr1 = np.hstack([Xtr2, np.ones((len(Xtr2),1))]); Xva1 = np.hstack([Xva2, np.ones((len(Xva2),1))])
    A = Xtr1.T@Xtr1 + alpha*np.eye(Xtr1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Xtr1.T@ytr)
    p = Xva1@w
    mae = np.abs(p-yva).mean()
    r2 = 1-((p-yva)**2).sum()/((yva-yva.mean())**2).sum()
    return mae, r2

print('\nconfig search (target val MAE ~61.109):')
for name,(Xa,Xb) in {'zero':(Xtr_raw,Xva_raw),'med':(Xtr_med,Xva_med)}.items():
    for std in [False,True]:
        for alpha in [0.0,1.0,10.0,100.0,1000.0]:
            mae,r2 = fit_eval(Xa,ytr,Xb,yva,alpha,std)
            print(f'impute={name:4s} std={std!s:5s} alpha={alpha:7.1f}  valMAE={mae:7.3f}  R2={r2:.4f}')

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
days = agent_api.snapshot_days()
tr_days = days['train']

df = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def ridge_fit_pred(Xtr, ytr, Xva, alpha=1.0, standardize=True):
    Xtr = np.asarray(Xtr, float); Xva = np.asarray(Xva, float)
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr),1))]); Xva1 = np.hstack([Xva, np.ones((len(Xva),1))])
    A = Xtr1.T@Xtr1 + alpha*np.eye(Xtr1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Xtr1.T@ytr)
    return Xva1@w, w

def proxy_mae(table, feat_subset=None, train_days=None, val_days=None, alpha=1.0, impute='zero'):
    d = table if feat_subset is None else table[['household_key','snapshot_day']+list(feat_subset)]
    f = [c for c in d.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(d, on=['household_key','snapshot_day'], how='inner')
    train_days = train_days or [x for x in tr_days if x not in val_days] if val_days else tr_days
    tr = m[m.snapshot_day.isin(train_days)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).copy(); Xva = va[f].astype(float).copy()
    if impute=='zero':
        Xtr=Xtr.fillna(0.0); Xva=Xva.fillna(0.0)
    else:
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    p,_ = ridge_fit_pred(Xtr.values, tr['future_spend_4w'].values, Xva.values, alpha=alpha)
    return np.abs(p - va['future_spend_4w'].values).mean()

# proxy split: train on 95..403, validate on 431 (mimics future-shift)
for alpha in [0.1, 1.0, 10.0, 100.0]:
    m = proxy_mae(t, val_days=[431], alpha=alpha)
    print(f'proxy val=431 alpha={alpha}: MAE {m:.3f}')
# also val on {403,431}
for alpha in [1.0, 10.0]:
    m = proxy_mae(t, val_days=[403,431], alpha=alpha)
    print(f'proxy val=403+431 alpha={alpha}: MAE {m:.3f}')

# residual pattern by snapshot day (in-sample-ish, train on 95..403 predict 431 and also per-day on train)
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
f = feats
tr = m[m.snapshot_day.isin([x for x in tr_days if x!=431])]
va = m[m.snapshot_day==431]
Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
p,_ = ridge_fit_pred(Xtr, tr['future_spend_4w'].values, Xva, alpha=1.0)
res = p - va['future_spend_4w'].values
print('\nday-431 proxy: MAE %.3f  mean pred %.1f  mean actual %.1f  bias %.2f' % (np.abs(res).mean(), p.mean(), va['future_spend_4w'].mean(), res.mean()))
# error by target decile
q = pd.qcut(va['future_spend_4w'].values, 10, duplicates='drop')
print(pd.DataFrame({'y':va['future_spend_4w'].values,'p':p,'q':q}).groupby('q', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g.y.mean(),'p':g.p.mean(),'mae':np.abs(g.p-g.y).mean()})).round(1))

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')

print('mean/zero-rate of target by snapshot day:')
g = m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda x:(x==0).mean(),'count'])
g.columns=['mean','median','zero_rate','n']; print(g.round(2))

# churn structure: rows with recent activity but zero future spend
m['spend84_0'] = m['spend_84']==0
print('\nby spend_84==0:')
print(m.groupby('spend84_0')['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

m['rec28'] = m['spend_28']==0
print('\nby spend_28==0:')
print(m.groupby('rec28')['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# zero-streak (consecutive trailing 28d zero blocks) vs target
print('\nby fwd28_zero_ct (count of trailing 28d blocks with zero spend, 0-6):')
print(m.groupby(m['fwd28_zero_ct'].fillna(6).astype(int))['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# recency buckets
print('\nby recency (days since last purchase):')
m['rec_b'] = pd.cut(m['recency'], [-1,7,14,28,56,112,10000])
print(m.groupby('rec_b', observed=True)['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# error decomposition of current best proxy model on day 431
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
trd = [d for d in agent_api.snapshot_days()['train'] if d!=431]
tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day==431]
Xtr = tr[feats].astype(float).fillna(0).values; Xva = va[feats].astype(float).fillna(0).values
mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
w = np.linalg.solve(A.T@A+1.0*np.eye(A.shape[1]), A.T@tr['future_spend_4w'].values)
p = B@w; y = va['future_spend_4w'].values
err = p-y
print('\nday431 proxy MAE %.2f | by y==0: n=%d, mae=%.1f, meanpred=%.1f | by y>0: mae=%.1f' % (
    np.abs(err).mean(), (y==0).sum(), np.abs(err[y==0]).mean(), p[y==0].mean(), np.abs(err[y>0]).mean()))
print('share of total abs err from y==0 rows: %.1f%%' % (100*np.abs(err[y==0]).sum()/np.abs(err).sum()))
print('share from y>top-decile: %.1f%%' % (100*np.abs(err[y>=np.quantile(y,0.9)]).sum()/np.abs(err).sum()))

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def regime_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value']]
    hh = view.households
    if isinstance(hh, pd.DataFrame): hh = list(hh.index)
    else: hh = list(pd.unique(pd.Series(np.asarray(hh).ravel())))
    out = pd.DataFrame(index=hh)
    # trailing window spends
    for w in [28,56,84,112,168,364]:
        s = tx[tx.day > day-w].groupby('household_key')['sales_value'].sum()
        out['spend_%d'%w] = s.reindex(hh).fillna(0.0)
    # weekly spends wk0..wk7
    for k in range(8):
        s = tx[(tx.day > day-7*(k+1)) & (tx.day <= day-7*k)].groupby('household_key')['sales_value'].sum()
        out['w%d'%k] = s.reindex(hh).fillna(0.0)
    out['spend_life'] = tx.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    out['recency'] = (day - tx.groupby('household_key')['day'].max()).reindex(hh)
    out['tenure'] = (day - tx.groupby('household_key')['day'].min()).reindex(hh)
    # consecutive trailing 28d zero blocks (most recent first)
    zk = np.zeros(len(hh))
    sp = {}
    for k in range(1,7):
        s = tx[(tx.day > day-28*k) & (tx.day <= day-28*(k-1))].groupby('household_key')['sales_value'].sum()
        sp[k] = s.reindex(hh).fillna(0.0).values
    zc = np.zeros(len(hh))
    for i in range(len(hh)):
        for k in range(1,7):
            if sp[k][i] <= 0: zc[i] += 1
            else: break
    out['zero_streak28'] = zc
    # churn flags
    out['z28'] = (out['spend_28']<=0).astype(float)
    out['z56'] = (out['spend_56']<=0).astype(float)
    out['z84'] = (out['spend_84']<=0).astype(float)
    # recency bucket one-hots
    r = out['recency'].values
    out['r_le7']   = (r<=7).astype(float)
    out['r_8_14']  = ((r>7)&(r<=14)).astype(float)
    out['r_15_28'] = ((r>14)&(r<=28)).astype(float)
    out['r_29_56'] = ((r>28)&(r<=56)).astype(float)
    out['r_57_112']= ((r>56)&(r<=112)).astype(float)
    out['r_gt112'] = (r>112).astype(float)
    # log1p compressions
    for c in ['spend_28','spend_56','spend_84','spend_112','spend_168','spend_364','spend_life',
              'w0','w1','w2','w3','wk_avg_4','wk_avg_8','wk_avg_84','fwd28_mean','fwd28_median',
              'fwd28_max','fwd28_min','spend_lag1','spend_lag2','x_s28','x_prev28',
              'x_life_rate_wk','spend_rate_life','x_b_mean_val','x_b_med_val']:
        if c in out.columns:
            out['L_'+c] = np.log1p(out[c].clip(lower=0))
    # regime interactions: log level x recency regime
    L84 = out['L_spend_84'].values
    for b in ['r_le7','r_8_14','r_15_28','r_29_56','r_57_112','r_gt112']:
        out['i_'+b] = out[b].values * L84
    out['i_z28_L84'] = out['z28'].values * L84
    out['i_z28_L28'] = out['z28'].values * out['L_spend_28'].values
    out['i_streak_L84'] = out['zero_streak28'].values * L84
    # trend-in-log
    out['L_ratio_28_84'] = np.log1p(out['spend_28']) - np.log1p(out['spend_84'])
    out['L_ratio_84_364'] = np.log1p(out['spend_84']) - np.log1p(out['spend_364'].clip(lower=0))
    return out.astype(float)

bf = agent_api.build_features(regime_feats)
print('built', bf.shape, 'snapshots', sorted(bf.snapshot_day.unique()))

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
newcols = [c for c in bf.columns if c not in ('household_key','snapshot_day')]
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

# proxy eval helper
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean(), np.abs(p-yva).mean()  # mae

base = proxy_mae(e008, [431]); print('proxy E008 val431: %.3f' % base[0])
c1 = proxy_mae(comb, [431]); print('proxy E008+regime val431: %.3f' % c1[0])
c2 = proxy_mae(comb, [403,431]); print('proxy E008+regime val403+431: %.3f' % c2[0])
b2 = proxy_mae(e008, [403,431]); print('proxy E008 val403+431: %.3f' % b2[0])

# also test: only logs+flags, no interactions
sub1 = e008.merge(bf[[c for c in newcols if not c.startswith('i_')]], on=['household_key','snapshot_day'])
print('proxy E008+logs/flags only val431: %.3f' % proxy_mae(sub1,[431])[0])

path = agent_api.save_table(comb, 'e009_regime.parquet')
print('saved', path)

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def analog_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value','basket_id']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    # compact feature builder for any anchor day <= day
    def build(anchor):
        t = tx[tx.day <= anchor]
        g = t.groupby('household_key')['sales_value']
        f = pd.DataFrame(index=hh)
        f['s28'] = t[t.day > anchor-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s84'] = t[t.day > anchor-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s364'] = t[t.day > anchor-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['rec'] = (anchor - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
        f['ten'] = (anchor - t.groupby('household_key')['day'].min()).reindex(hh).fillna(0)
        f['ntr28'] = t[t.day > anchor-28].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
        # zero streak in trailing 28d blocks
        sp = [t[(t.day > anchor-28*k) & (t.day <= anchor-28*(k-1))].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(1,5)]
        zs = np.zeros(len(hh))
        for i in range(len(hh)):
            for k in range(4):
                if sp[k][i] <= 0: zs[i] += 1
                else: break
        f['zs'] = zs
        return f
    # training rows: past snapshots with realized future 4w spend (needs anchor+28 <= day)
    rows, ys = [], []
    anchors = [day - 28*k for k in range(1, 13) if day - 28*k >= 95]
    for a in anchors:
        f = build(a)
        fut = tx[(tx.day > a) & (tx.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        rows.append(f); ys.append(fut.values)
    Xtr = pd.concat(rows); ytr = np.concatenate(ys)
    Xcur = build(day)
    # k-NN on log features
    cols = ['s28','s84','s364','rec','ten','ntr28','zs']
    Ltr = np.log1p(Xtr[cols].clip(lower=0).values)
    Lcu = np.log1p(Xcur[cols].clip(lower=0).values)
    mu,sd = Ltr.mean(0), Ltr.std(0); sd[sd==0]=1
    A = (Ltr-mu)/sd; B = (Lcu-mu)/sd
    yl = np.log1p(ytr)
    # subsample for speed
    rng = np.random.RandomState(0)
    idx = rng.choice(len(A), min(len(A), 12000), replace=False)
    A, yl_s = A[idx], yl[idx]
    # chunked k-NN, k=25
    k = 25
    preds = np.zeros(len(B))
    for st in range(0, len(B), 2000):
        en = min(st+2000, len(B))
        d = -2*A@B[st:en].T + (A*A).sum(1)[:,None]
        nn = np.argpartition(d, k, axis=0)[:k]
        preds[st:en] = np.expm1(yl_s[nn].mean(0))
    Xcur['knn28'] = preds
    # binned conditional mean: (s28 decile) x recency bucket
    qb = pd.qcut(np.log1p(Xtr['s28']), 10, labels=False, duplicates='drop')
    rb = pd.cut(Xtr['rec'], [-1,7,14,28,56,112,10**6], labels=False)
    key = pd.Series(qb.astype(str)+'_'+rb.astype(str))
    tab = pd.Series(yl).groupby(key.values).mean()
    kc = pd.Series(pd.qcut(np.log1p(Xcur['s28']), 10, labels=False, duplicates='drop').astype(str)+'_'+
                   pd.cut(Xcur['rec'], [-1,7,14,28,56,112,10**6], labels=False).astype(str))
    Xcur['binmean'] = kc.map(tab).astype(float).fillna(yl.mean()).values
    Xcur['log_knn'] = np.log1p(preds.clip(lower=0))
    Xcur['log_bin'] = np.log1p(Xcur['binmean'].clip(lower=0))
    return Xcur[['knn28','binmean','log_knn','log_bin']].astype(float)

bf = agent_api.build_features(analog_feats)
print('built', bf.shape)
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean()

print('proxy E008          val431: %.3f' % proxy_mae(e008,[431]))
print('proxy E008+analog   val431: %.3f' % proxy_mae(comb,[431]))
print('proxy E008          v403+431: %.3f' % proxy_mae(e008,[403,431]))
print('proxy E008+analog   v403+431: %.3f' % proxy_mae(comb,[403,431]))
# analog features alone
an = tt.merge(bf, on=['household_key','snapshot_day'])
print('proxy analog-only   val431: %.3f' % proxy_mae(bf,[431]))
# correlation with target
mm = tt.merge(bf, on=['household_key','snapshot_day'])
print('\ncorr with target:'); print(mm[['knn28','binmean','log_knn','log_bin','future_spend_4w']].corr()['future_spend_4w'].round(3))
agent_api.save_table(comb, 'e009_analog.parquet')

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def analog_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value','basket_id']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    def build(anchor):
        t = tx[tx.day <= anchor]
        f = pd.DataFrame(index=hh)
        f['s28'] = t[t.day > anchor-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s84'] = t[t.day > anchor-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s364'] = t[t.day > anchor-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['rec'] = (anchor - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
        f['ten'] = (anchor - t.groupby('household_key')['day'].min()).reindex(hh).fillna(0)
        f['ntr28'] = t[t.day > anchor-28].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
        sp = [t[(t.day > anchor-28*k) & (t.day <= anchor-28*(k-1))].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(1,5)]
        zs = np.zeros(len(hh))
        for i in range(len(hh)):
            for k in range(4):
                if sp[k][i] <= 0: zs[i] += 1
                else: break
        f['zs'] = zs
        return f
    rows, ys = [], []
    anchors = [day - 28*k for k in range(1, 13) if day - 28*k >= 56]
    for a in anchors:
        f = build(a)
        fut = tx[(tx.day > a) & (tx.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        rows.append(f); ys.append(fut.values)
    Xcur = build(day)
    cols = ['s28','s84','s364','rec','ten','ntr28','zs']
    if rows:
        Xtr = pd.concat(rows); ytr = np.concatenate(ys)
        yl = np.log1p(ytr)
        Ltr = np.log1p(Xtr[cols].clip(lower=0).values)
        Lcu = np.log1p(Xcur[cols].clip(lower=0).values)
        mu,sd = Ltr.mean(0), Ltr.std(0); sd[sd==0]=1
        A = (Ltr-mu)/sd; B = (Lcu-mu)/sd
        rng = np.random.RandomState(0)
        idx = rng.choice(len(A), min(len(A), 12000), replace=False)
        A, yl_s = A[idx], yl[idx]
        k = 25; preds = np.zeros(len(B))
        for st in range(0, len(B), 2000):
            en = min(st+2000, len(B))
            d = -2*A@B[st:en].T + (A*A).sum(1)[:,None]
            nn = np.argpartition(d, k, axis=0)[:k]
            preds[st:en] = np.expm1(yl_s[nn].mean(0))
        qb = pd.qcut(np.log1p(Xtr['s28']), 10, labels=False, duplicates='drop')
        rb = pd.cut(Xtr['rec'], [-1,7,14,28,56,112,10**6], labels=False)
        tab = pd.Series(yl).groupby((qb.astype(str)+'_'+rb.astype(str)).values).mean()
        kc = pd.Series(pd.qcut(np.log1p(Xcur['s28']), 10, labels=False, duplicates='drop').astype(str)+'_'+
                       pd.cut(Xcur['rec'], [-1,7,14,28,56,112,10**6], labels=False).astype(str))
        Xcur['binmean'] = kc.map(tab).astype(float).fillna(yl.mean()).values
    else:
        preds = np.zeros(len(hh)); Xcur['binmean'] = 0.0
    Xcur['knn28'] = preds
    Xcur['log_knn'] = np.log1p(preds.clip(lower=0))
    Xcur['log_bin'] = np.log1p(Xcur['binmean'].clip(lower=0))
    return Xcur[['knn28','binmean','log_knn','log_bin']].astype(float)

bf = agent_api.build_features(analog_feats)
print('built', bf.shape)
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean()

print('proxy E008          val431: %.3f' % proxy_mae(e008,[431]))
print('proxy E008+analog   val431: %.3f' % proxy_mae(comb,[431]))
print('proxy E008          v403+431: %.3f' % proxy_mae(e008,[403,431]))
print('proxy E008+analog   v403+431: %.3f' % proxy_mae(comb,[403,431]))
mm = tt.merge(bf, on=['household_key','snapshot_day'])
print('\ncorr with target:'); print(mm[['knn28','binmean','log_knn','log_bin','future_spend_4w']].corr()['future_spend_4w'].round(3))
path = agent_api.save_table(comb, 'e009_analog.parquet'); print('saved', path)

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def analog_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value','basket_id']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    def build(anchor):
        t = tx[tx.day <= anchor]
        f = pd.DataFrame(index=hh)
        f['s28'] = t[t.day > anchor-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s84'] = t[t.day > anchor-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s364'] = t[t.day > anchor-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['rec'] = (anchor - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
        f['ten'] = (anchor - t.groupby('household_key')['day'].min()).reindex(hh).fillna(0)
        f['ntr28'] = t[t.day > anchor-28].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
        sp = [t[(t.day > anchor-28*k) & (t.day <= anchor-28*(k-1))].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(1,5)]
        zs = np.zeros(len(hh))
        for i in range(len(hh)):
            for k in range(4):
                if sp[k][i] <= 0: zs[i] += 1
                else: break
        f['zs'] = zs
        return f
    rows, ys = [], []
    anchors = [day - 28*k for k in range(1, 13) if day - 28*k >= 56]
    for a in anchors:
        f = build(a)
        fut = tx[(tx.day > a) & (tx.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        rows.append(f); ys.append(fut.values)
    Xcur = build(day)
    cols = ['s28','s84','s364','rec','ten','ntr28','zs']
    if rows:
        Xtr = pd.concat(rows); ytr = np.concatenate(ys)
        yl = np.log1p(ytr)
        Ltr = np.log1p(Xtr[cols].clip(lower=0).values)
        Lcu = np.log1p(Xcur[cols].clip(lower=0).values)
        mu,sd = Ltr.mean(0), Ltr.std(0); sd[sd==0]=1
        A = (Ltr-mu)/sd; B = (Lcu-mu)/sd
        rng = np.random.RandomState(0)
        idx = rng.choice(len(A), min(len(A), 12000), replace=False)
        A, yl_s = A[idx], yl[idx]
        k = 25; preds = np.zeros(len(B))
        for st in range(0, len(B), 2000):
            en = min(st+2000, len(B))
            d = -2*A@B[st:en].T + (A*A).sum(1)[:,None]
            nn = np.argpartition(d, k, axis=0)[:k]
            preds[st:en] = np.expm1(yl_s[nn].mean(0))
        qb = pd.qcut(np.log1p(Xtr['s28']), 10, labels=False, duplicates='drop')
        rb = pd.cut(Xtr['rec'], [-1,7,14,28,56,112,10**6], labels=False)
        tab = pd.Series(yl).groupby((qb.astype(str)+'_'+rb.astype(str)).values).mean()
        kc = pd.Series(pd.qcut(np.log1p(Xcur['s28']), 10, labels=False, duplicates='drop').astype(str)+'_'+
                       pd.cut(Xcur['rec'], [-1,7,14,28,56,112,10**6], labels=False).astype(str))
        Xcur['binmean'] = kc.map(tab).astype(float).fillna(yl.mean()).values
    else:
        preds = np.zeros(len(hh)); Xcur['binmean'] = 0.0
    Xcur['knn28'] = preds
    Xcur['log_knn'] = np.log1p(np.clip(preds, 0, None))
    Xcur['log_bin'] = np.log1p(Xcur['binmean'].clip(lower=0))
    return Xcur[['knn28','binmean','log_knn','log_bin']].astype(float)

bf = agent_api.build_features(analog_feats)
print('built', bf.shape)
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean()

print('proxy E008          val431: %.3f' % proxy_mae(e008,[431]))
print('proxy E008+analog   val431: %.3f' % proxy_mae(comb,[431]))
print('proxy E008          v403+431: %.3f' % proxy_mae(e008,[403,431]))
print('proxy E008+analog   v403+431: %.3f' % proxy_mae(comb,[403,431]))
mm = tt.merge(bf, on=['household_key','snapshot_day'])
print('\ncorr with target:'); print(mm[['knn28','binmean','log_knn','log_bin','future_spend_4w']].corr()['future_spend_4w'].round(3))
path = agent_api.save_table(comb, 'e009_analog.parquet'); print('saved', path)

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e003 = agent_api.load_saved('e003_product_mix.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
eanalog = agent_api.load_saved('e009_analog.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean()

c03 = e008.merge(e003.drop(columns=[c for c in e003.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
call = c03.merge(eanalog.drop(columns=['knn28','binmean','log_knn','log_bin']), on=['household_key','snapshot_day'], how='inner') if False else c03

print('shapes: e008 %s e003 %s e002 %s' % (e008.shape, e003.shape, e002.shape))
print('proxy E008            v431: %.3f | v403+431: %.3f' % (proxy_mae(e008,[431]), proxy_mae(e008,[403,431])))
print('proxy E008+E003 mix   v431: %.3f | v403+431: %.3f' % (proxy_mae(c03,[431]), proxy_mae(c03,[403,431])))
print('proxy E008+E002 mkt   v431: %.3f | v403+431: %.3f' % (proxy_mae(c02,[431]), proxy_mae(c02,[403,431])))
c023 = c03.merge(e002.drop(columns=[c for c in e002.columns if c in c03.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('proxy E008+E003+E002  v431: %.3f | v403+431: %.3f' % (proxy_mae(c023,[431]), proxy_mae(c023,[403,431])))
# alpha sensitivity on best combo
for a in [0.3, 3.0, 30.0, 300.0]:
    print('proxy E008+E003+E002 alpha=%g v431: %.3f' % (a, proxy_mae(c023,[431],alpha=a)))
print('\ncols in e003 not in e008:', [c for c in e003.columns if c not in e008.columns][:40])
print('cols in e002 not in e008:', [c for c in e002.columns if c not in e008.columns][:40])

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def spline_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    out = pd.DataFrame(index=hh)
    s28 = tx[tx.day > day-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    s84 = tx[tx.day > day-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    s364 = tx[tx.day > day-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    rec = (day - tx.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
    # EWMA-style fwd mean proxy: use E008's fwd28_mean? Not available inside; rebuild as wk-based EWMA
    wks = [tx[(tx.day > day-7*(k+1)) & (tx.day <= day-7*k)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(8)]
    wks = np.array(wks)  # 8 x n
    ewma = np.zeros(len(hh)); w = 0.0
    for k in range(8):
        w = 0.7*w + (1 if k==0 else 0.3)
        ewma += w*wks[k]
    ewma = ewma / w
    out['ewma8'] = ewma
    out['L_s28'] = np.log1p(s28.values); out['L_s84'] = np.log1p(s84.values)
    out['L_s364'] = np.log1p(s364.values); out['L_ewma'] = np.log1p(np.clip(ewma,0,None))
    out['rec'] = rec.values
    # linear-spline bases (hinge functions) at fixed quantile knots of log-spend
    def hinges(col, nknots=8):
        x = out[col].values
        qs = np.quantile(x, np.linspace(0.05, 0.95, nknots))
        qs = np.unique(qs)
        for j,q in enumerate(qs):
            out['h_%s_%d'%(col,j)] = np.maximum(x-q, 0)
    for c in ['L_s28','L_s84','L_s364','L_ewma']:
        hinges(c, 8)
    # recency hinges (on raw days, capped)
    r = np.minimum(out['rec'].values, 200.0)
    for q in [3,7,10,14,21,28,42,56,84,112,150]:
        out['hr_%d'%q] = np.maximum(r-q, 0)
    # two-part lookup: P(y>0) and E[y|y>0] by (s28 decile x recency bucket) from past anchors
    rows_p, rows_l, keys_all = [], [], []
    anchors = [day-28*k for k in range(1,13) if day-28*k >= 56]
    def keyfn(s28v, recv, qs, rb_edges):
        qb = pd.Series(np.searchsorted(qs, np.log1p(np.clip(s28v,0,None)))).astype(str)
        rb = pd.Series(np.searchsorted(rb_edges, recv)).astype(str)
        return (qb+'_'+rb).values
    ps, ls = [], []
    for a in anchors:
        t = tx[tx.day <= a]
        sa = t[t.day > a-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ra = (a - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999).values
        ya = t[(t.day > a) & (t.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ps.append((ya>0).astype(float)); ls.append(ya)
        ps_key = None
    # build lookup from pooled anchors
    allkeys = []
    for i,a in enumerate(anchors):
        t = tx[tx.day <= a]
        sa = t[t.day > a-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        ra = (a - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999).values
        ya = t[(t.day > a) & (t.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values
        qs = np.quantile(np.log1p(sa), np.linspace(0,1,11))[1:-1]
        rbe = np.array([7,14,28,56,112])
        allkeys.append(pd.DataFrame({'k':keyfn(sa,ra,qs,rbe),'p':(ya>0).astype(float),'l':ya}))
    if allkeys:
        pool = pd.concat(allkeys)
        tab_p = pool.groupby('k')['p'].mean(); tab_l = pool.groupby('k')['l'].mean()
        gp = pool['p'].mean(); gl = pool['l'].mean()
        qs_cur = np.quantile(np.log1p(s28.values), np.linspace(0,1,11))[1:-1]
        kc = keyfn(s28.values, rec.values, qs_cur, np.array([7,14,28,56,112]))
        p_hat = pd.Series(kc).map(tab_p).astype(float).fillna(gp).values
        l_hat = pd.Series(kc).map(tab_l).astype(float).fillna(gl).values
        out['p_active'] = p_hat; out['lvl_given_active'] = l_hat
        out['pred_2p'] = p_hat * l_hat
        out['L_pred2p'] = np.log1p(out['pred_2p'].clip(lower=0))
    return out.astype(float)

bf = agent_api.build_features(spline_feats)
print('built', bf.shape, bf.columns.tolist()[:8], '...')
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
comb2 = comb.merge(e002.drop(columns=[c for c in e002.columns if c in comb.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('comb', comb.shape, 'comb2', comb2.shape)

tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def loo_mae(table, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    maes = []
    for d in tr_days:
        tr = m[m.snapshot_day != d]; va = m[m.snapshot_day == d]
        Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
        ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
        mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
        w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        p = B@w
        maes.append(np.abs(p-yva).mean())
    return np.mean(maes), maes

for name, tab in [('E008', e008), ('E008+spline2p', comb), ('E008+spline2p+mkt', comb2)]:
    for a in ([1.0, 100.0] if name!='E008' else [1.0, 100.0]):
        lm, _ = loo_mae(tab, alpha=a)
        print('LOO proxy %-18s alpha=%g: %.3f' % (name, a, lm))
path = agent_api.save_table(comb, 'e009_spline2p.parquet'); print('saved', path)

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
e009a = agent_api.load_saved('e009_analog.parquet')   # knn28, binmean, log_knn, log_bin
e009s = agent_api.load_saved('e009_spline2p.parquet') # includes p_active, lvl_given_active, pred_2p
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

def loo_mae(table, alpha=1.0, feats=None):
    f = feats or [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    maes = []
    for d in tr_days:
        tr = m[m.snapshot_day != d]; va = m[m.snapshot_day == d]
        Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
        ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
        mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
        w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        p = B@w
        maes.append(np.abs(p-yva).mean())
    return np.mean(maes)

def loo_single(col, table):
    # single-feature ridge (with intercept) ~ scaled fit
    return loo_mae(table, alpha=1.0, feats=[col])

print('standalone single-feature LOO MAE:')
for col in ['spend_84','spend_28','fwd28_mean','knn28','binmean','pred_2p','p_active','lvl_given_active','ewma8']:
    src = e009s if col in e009s.columns else (e009a if col in e009a.columns else e008)
    if col in src.columns:
        print('  %-18s %.3f' % (col, loo_single(col, src)))

print('\nLOO comparisons:')
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('E008            : %.3f' % loo_mae(e008, alpha=100))
print('E008+E002(mkt)  : %.3f' % loo_mae(c02, alpha=100))
# E008 + only the 4 analog features
c_a = e008.merge(e009a[['household_key','snapshot_day','knn28','binmean','log_knn','log_bin']], on=['household_key','snapshot_day'], how='inner')
print('E008+analog4    : %.3f' % loo_mae(c_a, alpha=100))
# E008 + only pred_2p family
c_p = e008.merge(e009s[['household_key','snapshot_day','p_active','lvl_given_active','pred_2p']], on=['household_key','snapshot_day'], how='inner')
print('E008+2p3        : %.3f' % loo_mae(c_p, alpha=100))
# curated small: top features + analog
top = ['spend_84','spend_112','spend_56','wk_avg_8','spend_168','fwd28_mean','x_life_rate_wk','spend_rate_life',
       'fwd28_median','spend_28','wk_avg_4','fwd28_max','fwd28_k1','spend_lag1','recency','tenure',
       'knn28','binmean','pred_2p','day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx']
cur = e009s[e009s.columns.intersection(['household_key','snapshot_day']+top)]
print('curated-small   : %.3f' % loo_mae(cur, alpha=100))
print('curated-small a1: %.3f' % loo_mae(cur, alpha=1))
# blend check: correlation of knn28 residual-with-E008-fit? approximate: partial corr
m = tt.merge(e009a, on=['household_key','snapshot_day']).merge(e008, on=['household_key','snapshot_day'], suffixes=('','_8'))
print('\nknn28 corr vs spend_84:', round(m['knn28'].corr(m['spend_84']),3))
print('pred_2p corr vs spend_84:', round(e009s[['pred_2p']].merge(tt,on=['household_key','snapshot_day'])['pred_2p'].corr(
    tt.merge(e008,on=['household_key','snapshot_day'])['spend_84']),3))

# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

# NaN structure
nanrate = e008.isna().mean().sort_values(ascending=False)
print('cols with NaN>0:'); print((nanrate[nanrate>0]*100).round(1).to_string())

def fwd_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(B@w - yva).mean()

# 1) day x level interactions
d = e008.copy()
for lev in ['spend_84','spend_28','L_s84','L_s28']:
    if lev in d.columns:
        d['di_x_'+lev] = d['day_idx']*d[lev]
    else:
        # L_s84 not in e008; approximate with log of spend_84
        d['L_s84'] = np.log1p(d['spend_84'].clip(lower=0)); d['L_s28'] = np.log1p(d['spend_28'].clip(lower=0))
        d['di_x_'+lev] = d['day_idx']*d[lev]
print('\n[1] day x level interactions:')
print('  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('  +dayxlevel  v431: %.3f' % fwd_mae(d,[431]))

# 2) marketing under forward proxies
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('\n[2] marketing block:')
print('  E008      v431: %.3f | v403: %.3f' % (fwd_mae(e008,[431]), fwd_mae(e008,[403])))
print('  +mkt      v431: %.3f | v403: %.3f' % (fwd_mae(c02,[431]), fwd_mae(c02,[403])))

# 4) winsorize heavy spend cols at per-snapshot p99
w = e008.copy()
for c in ['spend_84','spend_112','spend_56','spend_168','spend_28','spend_364','spend_life','fwd28_max','fwd28_mean','wk_avg_84','wk_avg_8']:
    if c in w.columns:
        cap = w.groupby('snapshot_day')[c].transform(lambda s: s.quantile(0.99))
        w[c+'_w'] = np.minimum(w[c].fillna(0), cap)
print('\n[4] winsorized top spends:')
print('  E008        v431: %.3f' % fwd_mae(e008,[431]))
print('  +winsor     v431: %.3f' % fwd_mae(w,[431]))

# 5) median imputation variant (proxy only)
def fwd_mae_med(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float); med = Xtr.median(); Xtr = Xtr.fillna(med).values
    Xva = va[f].astype(float).fillna(med).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    ww = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(B@ww - yva).mean()
print('\n[5] imputation: zero vs median (E008, v431): %.3f vs %.3f' % (fwd_mae(e008,[431]), fwd_mae_med(e008,[431])))

# ---- cell ----
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

# ---- cell ----
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