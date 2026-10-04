import agent_api, pandas as pd
for n in ['e018_basestab','rich_behavioral','e006_composition','e007_lagseq','e010_decay','e011_price','e013_te_clean','e014_dm','e016_peer','season','macro','mkt_demo','rfm28']:
    try:
        df = agent_api.load_saved(n + '.parquet')
        print('==', n, df.shape)
        print(list(df.columns))
        print()
    except Exception as e:
        print('==', n, 'ERR', repr(e))
print('snapshot days:', agent_api.snapshot_days())
v = agent_api.snapshot(459)
print('txn shape @459:', v.transactions.shape)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
for n in ['rich_behavioral','e006_composition','e007_lagseq','e010_decay','e011_price','e013_te_clean','e014_dm','e016_peer']:
    df = agent_api.load_saved(n + '.parquet')
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    print('==', n, len(cols))
    print(cols)
    print()

# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
train = df[df.future_spend_4w.notna()].copy()
val = df[df.future_spend_4w.isna()].copy()
print('train rows', len(train), 'val rows', len(val))

feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce')
med = num.median()
X = num.fillna(med).clip(-1e6,1e6)
mu, sd = X.iloc[train.index].mean(), X.iloc[train.index].std().replace(0,1)
# careful: use positional split
is_tr = df.future_spend_4w.notna().values
Z = ((X - X.mean()) / X.std().replace(0,1)).values
Ztr, ytr = Z[is_tr], df.future_spend_4w.values[is_tr]
Zva = Z[~is_tr]

def ridge_fit(Zt, y, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@y)
def mae(y, p): return np.abs(y-p).mean()

w = ridge_fit(Ztr, ytr, 5.0)
pv = Zva@w
print('ridge raw-y val MAE:', round(mae(df.future_spend_4w.values[~is_tr], pv),3))

# log-target variant
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
pv2 = np.expm1(Zva@w2)
print('ridge log-y val MAE:', round(mae(df.future_spend_4w.values[~is_tr], pv2),3))

ptr = Ztr@w
res = ytr - ptr
print('\ntarget describe:', pd.Series(ytr).describe())
print('train MAE:', round(mae(ytr,ptr),3))
tr_df = train.copy(); tr_df['pred']=ptr; tr_df['res']=res
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print('\ntrain MAE by spend28 quintile:')
print(tr_df.groupby('b', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g.future_spend_4w.mean(),'p':g.pred.mean(),'mae':g.res.abs().mean(),'bias':g.res.mean()})))
print('\nMAE by snapshot_day:')
print(tr_df.groupby('snapshot_day').apply(lambda g: pd.Series({'n':len(g),'mae':g.res.abs().mean(),'bias':g.res.mean(),'y':g.future_spend_4w.mean()})))
print('\nzero-target share:', (ytr==0).mean())
z = tr_df[tr_df.future_spend_4w==0]
print('zero-target rows: pred mean', round(z.pred.mean(),2), 'MAE', round(z.res.abs().mean(),2), 'n', len(z))
nz = tr_df[tr_df.future_spend_4w>0]
print('pos-target rows: MAE', round(nz.res.abs().mean(),2))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
bad = [c for c in feats if num[c].abs().max()>1e12 or num[c].isna().all()]
print('bad cols:', bad[:20], len(bad))
num = num[[c for c in feats if c not in bad]]
X = num.fillna(num.median())
is_tr = df.future_spend_4w.notna().values
y = df.future_spend_4w.values
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5).values
Ztr, Zva, ytr = Z[is_tr], Z[~is_tr], y[is_tr]

def ridge_fit(Zt, yv, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yv)
def mae(a,b): return np.abs(a-b).mean()

w = ridge_fit(Ztr, ytr, 5.0)
print('ridge raw-y val MAE:', round(mae(y[~is_tr], Zva@w),3))
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
print('ridge log-y val MAE:', round(mae(y[~is_tr], np.expm1(Zva@w2)),3))

ptr = Ztr@w
tr_df = df[is_tr].copy(); tr_df['pred']=ptr; tr_df['res']=ytr-ptr
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print('\ntrain MAE by spend28 quintile:')
print(tr_df.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nMAE by snapshot_day:')
print(tr_df.groupby('snapshot_day').agg(n=('res','size'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero-target share:', (ytr==0).mean())
z = tr_df[tr_df.future_spend_4w==0]
print('zero rows: pred mean', round(z.pred.mean(),2), 'MAE', round(z.res.abs().mean(),2), 'n', len(z))
print('pos rows MAE:', round(tr_df[tr_df.future_spend_4w>0].res.abs().mean(),2))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
allnan = [c for c in num.columns if num[c].notna().sum()==0]
print('all-nan cols:', allnan)
num = num.drop(columns=allnan)
X = num.fillna(num.median())
is_tr = df.future_spend_4w.notna().values
y = df.future_spend_4w.values
print('NaN in X:', X.isna().sum().sum())
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5)
print('NaN in Z:', Z.isna().sum().sum(), 'inf:', np.isinf(Z.values).sum())
Ztr, Zva, ytr = Z.values[is_tr], Z.values[~is_tr], y[is_tr]
def mae(a,b): return np.abs(a-b).mean()
def ridge_fit(Zt, yv, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yv)
w = ridge_fit(Ztr, ytr, 5.0)
pv = Zva@w
print('nan in pv:', np.isnan(pv).sum())
print('ridge raw-y val MAE:', round(mae(y[~is_tr], pv),3))
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
print('ridge log-y val MAE:', round(mae(y[~is_tr], np.expm1(Zva@w2)),3))
ptr = Ztr@w
tr_df = df[is_tr].copy(); tr_df['pred']=ptr; tr_df['res']=ytr-ptr
print('train MAE:', round(mae(ytr,ptr),3), 'bias:', round((ytr-ptr).mean(),3))
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print(tr_df.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nby snapshot_day:')
print(tr_df.groupby('snapshot_day').agg(n=('res','size'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero rows: pred mean', round(tr_df[tr_df.future_spend_4w==0].pred.mean(),2), 'MAE', round(tr_df[tr_df.future_spend_4w==0].res.abs().mean(),2))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
X = num.fillna(num.median())
y = df.future_spend_4w.values
is_tr = df.future_spend_4w.notna().values
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5)
Z['intercept']=1.0
Zv = Z.values

# pseudo-val: train on snapshot days <= 403, eval on 431
tr_mask = is_tr & (df.snapshot_day<=403).values
va_mask = is_tr & (df.snapshot_day==431).values
def fit(mask, lam=5.0, log=False):
    Zt, yt = Zv[mask], (np.log1p(y) if log else y)[mask]
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yt)
def mae(a,b): return np.abs(a-b).mean()
w = fit(tr_mask)
ptr, pva = Zv[tr_mask]@w, Zv[va_mask]@w
print('Pseudo-val (day 431) MAE:', round(mae(y[va_mask], pva),3))
print('train(<=403) MAE:', round(mae(y[tr_mask], ptr),3))
res = y[tr_mask]-ptr
t = df[tr_mask].copy(); t['pred']=ptr; t['res']=res
print('bias:', round(res.mean(),2))
t['b'] = pd.qcut(t.spend28, 5, duplicates='drop')
print(t.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero-target rows: n', (t.future_spend_4w==0).sum(), 'pred mean', round(t[t.future_spend_4w==0].pred.mean(),2))
v = df[va_mask].copy(); v['pred']=pva; v['res']=y[va_mask]-pva
print('\nday431: bias', round(v.res.mean(),2), 'MAE', round(v.res.abs().mean(),2))
# top ridge coefficients
coef = pd.Series(w[:-1], index=num.columns)
print('\nTop +coef:', coef.nlargest(12).round(2).to_dict())
print('Top -coef:', coef.nsmallest(12).round(2).to_dict())
# log-target variant with smearing
w2 = fit(tr_mask, log=True)
pva2 = np.expm1(Zv[va_mask]@w2)
ptr2 = np.expm1(Zv[tr_mask]@w2)
smear = (y[tr_mask]/np.clip(ptr2,1e-3,None)).mean()
print('\nlog-target pseudo-val MAE:', round(mae(y[va_mask], pva2),3), 'with smear:', round(mae(y[va_mask], pva2*smear),3))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
X = num.fillna(num.median())
y = df.future_spend_4w.values
is_tr = df.future_spend_4w.notna().values
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5); Z['intercept']=1.0
Zv = Z.values
def mae(a,b): return np.abs(a-b).mean()
def fit(mask, lam=5.0):
    Zt = Zv[mask]; yt = y[mask]
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yt)

days = sorted(df.snapshot_day[is_tr].unique())
print('OOF bias/MAE by eval day (train on all earlier days):')
for d in days[3:]:
    trm = is_tr & (df.snapshot_day < d).values
    vam = is_tr & (df.snapshot_day == d).values
    w = fit(trm)
    p = Zv[vam]@w
    r = y[vam]-p
    print(f"day {d}: n={vam.sum():4d} y_mean={y[vam].mean():6.1f} p_mean={p.mean():6.1f} bias={r.mean():6.1f} MAE={np.abs(r).mean():6.1f} zero_share={(y[vam]==0).mean():.3f}")

# zero vs nonzero target feature comparison (day 431 OOF preds)
d = 431
trm = is_tr & (df.snapshot_day < d).values
vam = is_tr & (df.snapshot_day == d).values
w = fit(trm)
t = df[vam].copy(); t['pred']=Zv[vam]@w
z, nz = t[t.future_spend_4w==0], t[t.future_spend_4w>0]
print('\nzero rows:', len(z), 'mean pred', round(z.pred.mean(),1), '| nonzero rows:', len(nz), 'mean pred', round(nz.pred.mean(),1))
key = ['spend28','spend7','recency','lag_zero_streak','lag1_is_zero','zero28','lag_nonzero_cnt_1_13','trend_7_28','trend_28_56','dec_spend14','trips28','actdays28','tenure','ratio28_364','active_weeks112']
print('\nfeature means: zero-target vs nonzero-target (day 431):')
cmp = pd.DataFrame({'zero': z[key].mean(), 'nonzero': nz[key].mean()})
cmp['ratio'] = (cmp.zero/cmp.nonzero).round(2)
print(cmp.round(1))
# how many zero-target rows have recency <= 14?
print('\nzero-target rows by recency bucket:')
print(pd.cut(z.recency, [-1,7,14,21,28,56,1000]).value_counts().sort_index())
print('\nnonzero-target rows by recency bucket:')
print(pd.cut(nz.recency, [-1,7,14,21,28,56,1000]).value_counts().sort_index())

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
print('macro cols not in e018:', [c for c in mac.columns if c not in df.columns])

def prep(frame, drop_allnan=True):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    if drop_allnan:
        num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

eval_table(df, 'e018 baseline')
dm = df.merge(mac.drop(columns=[c for c in mac.columns if c in df.columns]), on=['household_key','snapshot_day'], how='left')
eval_table(dm, 'e018 + all macro cols')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet')
if 'household_key' not in mac.columns: mac = mac.reset_index()
print('macro index/cols:', mac.columns.tolist()[:4], '... nunique days:', mac.snapshot_day.nunique())
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

eval_table(df, 'e018 baseline')
dm = df.merge(mac[[c for c in mac.columns if c not in df.columns] + ['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='left')
eval_table(dm, 'e018 + macro day-level cols')

# zero-target prototype: recency-based zero-risk gating
d0 = df.copy()
d0['zero_risk'] = np.clip((d0.recency-7)/21, 0, 1)
eval_table(d0.assign(zr_pred=d0.spend28*0), 'sanity')  # noop
d1 = df.copy()
d1['zero_risk'] = np.clip((d1.recency-7)/21, 0, 1)
eval_table(d1, 'e018 + zero_risk(recency)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

mac = agent_api.load_saved('macro.parquet').reset_index()
tt = agent_api.train_targets()
df = agent_api.load_saved('e018_basestab.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')

# 1) macro values by day
md = mac.groupby('snapshot_day').mean(numeric_only=True)
print('macro by day:')
print(md.round(3).to_string())

# 2) day-level future spend (train days only; view capped at 459)
v = agent_api.snapshot(459)
txn = v.transactions
rows=[]
for d in sorted(df.snapshot_day.unique()):
    if d>431: continue
    fut = txn[(txn.day>d)&(txn.day<=d+28)]
    rows.append({'day':d, 'fut_per_hh': fut.sales_value.sum()/len(fut.household_key.unique()) if len(fut) else 0})
fut = pd.DataFrame(rows).set_index('day')
print('\nday-level future spend per hh:')
print(fut.round(2).to_string())
j = md.join(fut)
print('\ncorrelation of macro features with day-level future spend:')
print(j.corr()['fut_per_hh'].drop('fut_per_hh').round(3).sort_values())

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

# 1) recency-interaction (household-level, no leakage)
d1 = df.copy()
d1['rec_g'] = np.clip((d1.recency-7)/21, 0, 1)
for c in ['spend28','spend7','spend56','spend112','dec_spend14','rwspend84','lag_spend_1','lag_mean_1_4','spend_yoy28','lt_spend']:
    d1[c+'_g'] = d1[c]*d1.rec_g
eval_table(d1, 'e018 + recency-gated spend interactions')

# 2) recency-based zero-risk gating (raw feature)
d2 = df.copy()
d2['zero_risk'] = np.clip((d2.recency-7)/21, 0, 1)
d2['zero_risk2'] = np.clip((d2.recency-14)/28, 0, 1)
d2['zero_risk3'] = np.clip((d2.recency-7)/14, 0, 1)
eval_table(d2, 'e018 + zero_risk raw (3 variants)')

# 3) log-transform heavy-tailed spend features
sp_cols = [c for c in df.columns if c.startswith('spend') or c.startswith('lag_spend') or c.startswith('dec_') or c in ('lt_spend','rwspend84','own_ly_spend4w')]
d3 = df.copy()
for c in sp_cols:
    d3[c+'_lg'] = np.log1p(d3[c].clip(lower=0))
eval_table(d3, 'e018 + log1p(spend cols)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))
    return Zv, is_tr, y

# A) seasonality: week-of-year sin/cos at FUTURE window midpoint
d = df.copy()
mid = d.snapshot_day + 14
wk = (mid+8)//7
d['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18)
d['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
d['fut_wk'] = wk
eval_table(d, 'e018 + future-week sin/cos')

# B) seasonality: retailer-wide future-window spend per active household, computed from txn history (leakage-safe: uses only past txn)
v = agent_api.snapshot(459)
txn = v.transactions
tot = txn.groupby('day').sales_value.sum()
d['fut_win_daily'] = [tot.iloc[min(int(t)+1, len(tot)-1):min(int(t)+28, len(tot)-1)].sum() for t in d.snapshot_day]
eval_table(d, 'e018 + future-window retailer daily total (diag only)')

# C) day-level target encoding of snapshot_day (strictly prior days only)
d = df.copy()
day_stats = df[df.future_spend_4w.notna()].groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
gm = df[df.future_spend_4w.notna()].future_spend_4w.mean()
d['te_day_mean'] = [day_stats.loc[day_stats.index<day, 'mean'].mean() if (day_stats.index<day).any() else gm for day in d.snapshot_day]
d['te_day_last'] = [day_stats.loc[day_stats.index<day, 'mean'].iloc[-1] if (day_stats.index<day).any() else gm for day in d.snapshot_day]
d['te_day_med'] = [day_stats.loc[day_stats.index<day, 'median'].mean() if (day_stats.index<day).any() else gm for day in d.snapshot_day]
eval_table(d, 'e018 + prior-day TE (mean/last/median)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet').reset_index()
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

# scaled macro features (leakage-safe: all trailing windows)
m = mac.set_index('snapshot_day')
d = df.copy()
d['m28'] = (d.snapshot_day.map(m.macro_spend28)/1e5)
d['m28_p'] = (d.snapshot_day.map(m.macro_spend_p28)/1e5)
d['m112'] = (d.snapshot_day.map(m.macro_spend112)/1e5)
d['m_growth'] = d.snapshot_day.map(m.macro_growth)
d['m_wk3'] = d.snapshot_day.map(m.macro_wk_ratio_last3)
d['m_hh28'] = d.snapshot_day.map(m.macro_hh28)/1e3
d['m_per_hh28'] = d.snapshot_day.map(m.macro_spend_per_hh28)
eval_table(d, 'e018 + scaled macro (7)')

# only the ratio/trend ones (scale-free)
d2 = df.copy()
d2['m_growth'] = d2.snapshot_day.map(m.macro_growth)
d2['m_wk3'] = d2.snapshot_day.map(m.macro_wk_ratio_last3)
d2['m_per_hh28'] = d2.snapshot_day.map(m.macro_spend_per_hh28)
d2['m_per_hh28_z'] = d2.m_per_hh28 - d2.m_per_hh28.mean()
eval_table(d2, 'e018 + macro ratios only (4)')

# macro + future-week sin/cos
d3 = d.copy()
wk = (d.snapshot_day+14+8)//7
d3['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18); d3['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
eval_table(d3, 'e018 + scaled macro + fut-week sin/cos')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet').reset_index()
mday = mac.groupby('snapshot_day').first()
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

d = df.copy()
d['m28'] = d.snapshot_day.map(mday.macro_spend28)/1e5
d['m28_p'] = d.snapshot_day.map(mday.macro_spend_p28)/1e5
d['m112'] = d.snapshot_day.map(mday.macro_spend112)/1e5
d['m_growth'] = d.snapshot_day.map(mday.macro_growth)
d['m_wk3'] = d.snapshot_day.map(mday.macro_wk_ratio_last3)
d['m_hh28'] = d.snapshot_day.map(mday.macro_hh28)/1e3
d['m_per_hh28'] = d.snapshot_day.map(mday.macro_spend_per_hh28)
eval_table(d, 'e018 + scaled macro (7)')

d2 = df.copy()
d2['m_growth'] = d2.snapshot_day.map(mday.macro_growth)
d2['m_wk3'] = d2.snapshot_day.map(mday.macro_wk_ratio_last3)
d2['m_per_hh28'] = d2.snapshot_day.map(mday.macro_spend_per_hh28)
eval_table(d2, 'e018 + macro ratios only (3)')

d3 = d.copy()
wk = (d3.snapshot_day+14+8)//7
d3['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18); d3['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
eval_table(d3, 'e018 + scaled macro + fut-week sin/cos')

# ---- cell ----
import agent_api, numpy as np, pandas as pd

def fn(view, snapshot_day):
    base = agent_api.load_saved('e018_basestab.parquet')
    df = base[base.snapshot_day == snapshot_day].copy()
    mac = agent_api.load_saved('macro.parquet').reset_index()
    mday = mac.groupby('snapshot_day').first()
    row = mday.loc[snapshot_day] if snapshot_day in mday.index else None
    def g(c):
        return float(row[c]) if row is not None and pd.notna(row[c]) else np.nan
    df['m28']        = g('macro_spend28')/1e5
    df['m28_p']      = g('macro_spend_p28')/1e5
    df['m28_ly']     = g('macro_spend28_ly')/1e5
    df['m112']       = g('macro_spend112')/1e5
    df['m_hh28']     = g('macro_hh28')/1e3
    df['m_per_hh28'] = g('macro_spend_per_hh28')
    df['m_per_hh28_ly'] = g('macro_spend_per_hh28_ly')
    df['m_growth']   = g('macro_growth')
    df['m_wk3']      = g('macro_wk_ratio_last3')
    df['m_ratio_ly'] = g('macro_ratio_ly')
    wk = (snapshot_day + 14 + 8) // 7
    df['fut_wk'] = float(wk)
    df['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18)
    df['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    return df.set_index('household_key')[cols]

out = agent_api.build_features(fn)
print(out.shape)
print(out.groupby('snapshot_day')[['m28','m_growth','m_wk3','m_per_hh28','fut_wk']].first())
path = agent_api.save_table(out, 'e019_macroctx.parquet')
print(path)