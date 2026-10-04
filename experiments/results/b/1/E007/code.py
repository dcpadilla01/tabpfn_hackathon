import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e001_history.parquet')
print('shape', t.shape)
print('cols', t.columns.tolist())
print(t.dtypes.to_string())
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero share', (tt.future_spend_4w==0).mean())
print('rows per snapshot (train):')
print(tt.groupby('snapshot_day').size())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)

num = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w','index')]
cat = [c for c in df.columns if df[c].dtype.name=='category']
print('num', len(num), 'cat', cat)

# correlation with target
for c in num:
    print(f"{c:20s} corr={df[c].corr(df.future_spend_4w): .3f}  nan={df[c].isna().mean():.2f}")

# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w

def mae(p): return np.abs(p - y).mean()
print('mean pred', mae(np.full(len(y), y.mean())))
print('median pred', mae(np.full(len(y), y.median())))
for c in ['spend_7','spend_28','spend_56','spend_84','spend_182','spend_365','spend_all','weekly_rate_84','spend_prev28']:
    print(f'persist {c:15s} MAE={mae(df[c].fillna(0).values):8.3f}')

# blend of spends via ridge
from numpy.linalg import lstsq
X = df[['spend_7','spend_28','spend_56','spend_84','spend_182','spend_365']].fillna(0).values
w = lstsq(X, y.values, rcond=None)[0]
print('ridge-ish coefs', np.round(w,3), 'MAE', mae(X@w))

# ratio target/spend_84
r = y / df.spend_84.replace(0, np.nan)
print('ratio quantiles', r.quantile([.1,.25,.5,.75,.9]).round(3).to_dict())

# binned: target vs spend_84
b = pd.qcut(df.spend_84, 10, duplicates='drop')
g = df.groupby(b, observed=True).agg(y=('future_spend_4w','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
print(g.round(1))

# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(431)
print('day', v.day, 'week', v.week)
hh = v.households
print('households type', type(hh), hh.shape if hasattr(hh,'shape') else len(hh))
print(hh.head())
tx = v.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print('demographics', v.demographics.shape)
print('n households with tx', tx.household_key.nunique())
sd = agent_api.snapshot_days(); print(sd)
print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)

# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(431)
hh = v.households
print(type(hh), hh)
tx = v.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print('demographics', v.demographics.shape)
print('n hh with tx', tx.household_key.nunique())
print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)
print('snapshot_days', agent_api.snapshot_days())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')

def onehot(df, cats):
    parts = []
    for c in cats:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        parts.append(d.values.astype(float))
    return parts

def fit_ridge(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd
    Z = np.hstack([Z, np.ones((len(Z),1))])
    A = Z.T@Z + lam*np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T@ytr), mu, sd

def predict(model, Xte):
    w, mu, sd = model
    Z = (Xte-mu)/sd
    Z = np.hstack([Z, np.ones((len(Z),1))])
    return Z@w

def prep(df, num_cols, cats, target=True):
    Xnum = df[num_cols].astype(float).values.copy()
    Xnum[np.isnan(Xnum)] = 0.0
    parts = [Xnum] + onehot(df, cats)
    X = np.hstack(parts)
    y = df[target_col].values if target else None
    return X, y

target_col='future_spend_4w'
cats = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']
num_e000 = ['has_demographics','snapshot_day_index','week_of_year']
num_e001 = [c for c in t.columns if t[c].dtype.kind in 'if' and c not in ('household_key','snapshot_day','index')]

for name, num in [('E000', num_e000), ('E001', num_e001)]:
    X, y = prep(df, num, cats)
    tr = df.snapshot_day <= 403
    va = df.snapshot_day == 431
    m = fit_ridge(X[tr.values], y[tr.values], lam=10.0)
    p = predict(m, X[va.values])
    print(f'{name}: local holdout(431) MAE={np.abs(p-y[va.values]).mean():.3f}  (harness: {"92.531" if name=="E000" else "63.025"})')

# ---- cell ----
import agent_api, pandas as pd, numpy as np

# Prototype new features at snapshot d=431 (outside build_features, capped view is fine for prototyping)
d = 431
v = agent_api.snapshot(d)
tx = v.transactions
t = agent_api.load_saved('e001_history.parquet')
hh = t.loc[t.snapshot_day==d, 'household_key'].values
print('n hh', len(hh))

tx = tx[tx.household_key.isin(set(hh))]
S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
days = np.arange(1, d+1)
S = S.reindex(columns=days, fill_value=0.0).reindex(hh, fill_value=0.0)
M = S.values.astype(np.float64)
C = np.zeros((M.shape[0], d+1)); C[:,1:] = M.cumsum(1)
idx = {h:i for i,h in enumerate(S.index)}

def spend(a, b):
    a = max(a,1); b = min(b,d)
    if a>b: return np.zeros(M.shape[0])
    return C[:,b]-C[:,a-1]

def row(h): return idx[h]

# windows j: [d-28j, d-28j+27]
sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])  # 13 x n
tj = np.stack([spend(d-28*j-84, d-28*j-1) for j in range(1,9)])
w = 0.5**np.arange(13)[:,None]  # 13 x 1
act = (sj>0).astype(float)
p_active13 = (w*act).sum(0)/w.sum()
p_active6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
usual_active = (w*sj*act).sum(0)/np.maximum(w*act).sum(0) if False else (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
exp_dec = (w*sj).sum(0)/w.sum()
ratios = np.clip(sj[:8]/(tj+10.0), 0, 3)
persist = (0.5**np.arange(8)[:,None]*ratios).sum(0)/0.5**np.arange(8).sum()
# streak
streak = np.zeros(M.shape[0])
for j in range(13):
    streak += act[j]*(streak>=j)
# hazard: P(active j | active j-1), j=2..13
num = (act[1:]*act[:-1]).sum(0); den = act[:-1].sum(0)
hazard = num/np.maximum(den,1e-9)
# cv
m13 = sj.mean(0); sd13 = sj.std(0); cv13 = sd13/(m13+1.0)
# last-year window [d-363, d-336]
ly = spend(d-363, d-336); ly_avail = (d>=364).astype(float)*np.ones(M.shape[0])
ly_ratio = ly/(sj[0]+10.0)
# gaps within [d-364, d]: use active days
Act = (M>0)
lastd = np.array([np.max(np.where(Act[i])[0])+1 if Act[i].any() else 0 for i in range(M.shape[0])])
# longest gap in last 364 days
gap_long = np.zeros(M.shape[0]); gap_n21 = np.zeros(M.shape[0])
for i in range(M.shape[0]):
    ad = np.where(Act[i])[0]
    ad = ad[(ad>=d-364)]
    if len(ad)==0: continue
    gaps = np.diff(np.concatenate([[-1], ad, [d]])) - 1  # gaps between active days incl edges
    gap_long[i] = gaps.max(); gap_n21[i] = (gaps>=21).sum()

out = pd.DataFrame({
 'p_active13':p_active13,'p_active6':p_active6,'usual_active':usual_active,'exp_dec':exp_dec,
 'persist':persist,'streak':streak,'hazard':hazard,'cv13':cv13,'ly_spend':ly,'ly_ratio':ly_ratio,
 'gap_long':gap_long,'gap_n21':gap_n21,'expected':p_active13*usual_active}, index=S.index)
print(out.describe().round(3).T[['mean','std','min','50%','max']])
print('corr with target:')
tt = agent_api.train_targets()
mg = out.reset_index().rename(columns={'index':'household_key'})
mg['snapshot_day']=d
m2 = mg.merge(tt[tt.snapshot_day==d], on=['household_key','snapshot_day'])
print(m2.drop(columns=['household_key','snapshot_day']).corr()['future_spend_4w'].round(3))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

d = 431
v = agent_api.snapshot(d)
tx = v.transactions
t = agent_api.load_saved('e001_history.parquet')
hh = t.loc[t.snapshot_day==d, 'household_key'].values
tx = tx[tx.household_key.isin(set(hh))]
S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
S = S.reindex(columns=np.arange(1,d+1), fill_value=0.0).reindex(hh, fill_value=0.0)
M = S.values.astype(np.float64)
C = np.zeros((M.shape[0], d+1)); C[:,1:] = M.cumsum(1)

def spend(a, b):
    a = max(a,1); b = min(b,d)
    if a>b: return np.zeros(M.shape[0])
    return C[:,b]-C[:,a-1]

sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])
tj = np.stack([spend(d-28*j-84, d-28*j-1) for j in range(1,9)])
w = 0.5**np.arange(13)[:,None]
act = (sj>0).astype(float)
p_active13 = (w*act).sum(0)/w.sum()
p_active6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
usual_active = (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
exp_dec = (w*sj).sum(0)/w.sum()
ratios = np.clip(sj[:8]/(tj+10.0), 0, 3)
persist = (0.5**np.arange(8)[:,None]*ratios).sum(0)/0.5**np.arange(8).sum()
streak = np.zeros(M.shape[0])
for j in range(13):
    streak += act[j]*(streak>=j)
num = (act[1:]*act[:-1]).sum(0); den = act[:-1].sum(0)
hazard = num/np.maximum(den,1e-9)
m13 = sj.mean(0); sd13 = sj.std(0); cv13 = sd13/(m13+1.0)
ly = spend(d-363, d-336); ly_avail = np.full(M.shape[0], float(d>=364))
ly_ratio = ly/(sj[0]+10.0)
Act = M>0
lastd = np.array([np.max(np.where(Act[i])[0])+1 if Act[i].any() else 0 for i in range(M.shape[0])])
gap_long = np.zeros(M.shape[0]); gap_n21 = np.zeros(M.shape[0])
for i in range(M.shape[0]):
    ad = np.where(Act[i])[0]
    ad = ad[ad>=d-364]
    if len(ad)==0: continue
    gaps = np.diff(np.concatenate([[-1], ad, [d]])) - 1
    gap_long[i] = gaps.max(); gap_n21[i] = (gaps>=21).sum()

out = pd.DataFrame({'p_active13':p_active13,'p_active6':p_active6,'usual_active':usual_active,'exp_dec':exp_dec,
 'persist':persist,'streak':streak,'hazard':hazard,'cv13':cv13,'ly_spend':ly,'ly_ratio':ly_ratio,
 'gap_long':gap_long,'gap_n21':gap_n21,'expected':p_active13*usual_active}, index=S.index)
print(out.describe().round(3).T[['mean','std','50%','max']])
tt = agent_api.train_targets()
mg = out.reset_index().rename(columns={'index':'household_key'}); mg['snapshot_day']=d
m2 = mg.merge(tt[tt.snapshot_day==d], on=['household_key','snapshot_day'])
print(m2.drop(columns=['household_key','snapshot_day']).corr()['future_spend_4w'].round(3).to_string())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

# Hypothesis E007: future 4w spend = P(active in next 4w) x usual spend when active.
# Explicit decomposition: activity probability over trailing 28d windows, usual per-active-window
# spend, streak, hazard, gaps, last-year same window.

def compute_new(view, d, hh_keys):
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh_keys))]
    S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
    S = S.reindex(columns=np.arange(1,d+1), fill_value=0.0).reindex(hh_keys, fill_value=0.0)
    M = S.values.astype(np.float64)
    n = M.shape[0]
    C = np.zeros((n, d+1)); C[:,1:] = M.cumsum(1)
    def spend(a,b):
        a=max(a,1); b=min(b,d)
        if a>b: return np.zeros(n)
        return C[:,b]-C[:,a-1]
    sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])
    act = (sj>0).astype(float)
    w = 0.5**np.arange(13)[:,None]
    p13 = (w*act).sum(0)/w.sum()
    p6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
    p3 = act[:3].mean(0)
    usual = (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
    expected = p13*usual
    streak = np.zeros(n)
    for j in range(13): streak += act*j*(streak>=j)  # placeholder fixed below
    streak = np.zeros(n)
    for j in range(13): streak += act[j]*(streak>=j)
    hazard = (act[1:]*act[:-1]).sum(0)/np.maximum(act[:-1].sum(0),1e-9)
    cv13 = sj.std(0)/(sj.mean(0)+1.0)
    Act = M>0
    gap_long = np.zeros(n); gap_n21 = np.zeros(n)
    for i in range(n):
        ad = np.where(Act[i])[0]; ad = ad[ad>=d-364]
        if len(ad)==0: continue
        gaps = np.diff(np.concatenate([[-1], ad, [d]]))-1
        gap_long[i]=gaps.max(); gap_n21[i]=(gaps>=21).sum()
    ly = spend(d-363, d-336) if d>=364 else np.zeros(n)
    ly_ratio = ly/(sj[0]+10.0)
    rvu = sj[0]/(usual+10.0)
    out = pd.DataFrame({
        'p_active13':p13,'p_active6':p6,'p_act3':p3,'usual_active':usual,'expected':expected,
        'log_expected':np.log1p(expected),'streak':streak,'hazard':hazard,'cv13':cv13,
        'gap_long':gap_long,'gap_n21':gap_n21,'ly_spend':ly,'ly_ratio':ly_ratio,'recent_vs_usual':rvu},
        index=pd.Index(hh_keys, name='household_key'))
    return out

def fn(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    if hh is None:
        tx = view.transactions
        fd = tx.groupby('household_key').day.min()
        hh = fd[fd <= d-84].index.values
    else:
        hh = hh.household_key.values if hasattr(hh,'household_key') else np.asarray(hh)
    return compute_new(view, d, hh)

F = agent_api.build_features(fn)
print('build ok', F.shape)
print(F.columns.tolist())
e1 = agent_api.load_saved('e001_history.parquet')
for d in [95, 207, 431, 543]:
    print(d, len(F[F.snapshot_day==d]), len(e1[e1.snapshot_day==d]))
agent_api.save_table(F, 'e007_new.parquet')

# ---- cell ----
import agent_api, pandas as pd, numpy as np

# Hypothesis E007: future 4w spend = P(active in next 4w) x usual spend when active.
def compute_new(view, d, hh_keys):
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh_keys))]
    S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
    S = S.reindex(columns=np.arange(1,d+1), fill_value=0.0).reindex(hh_keys, fill_value=0.0)
    M = S.values.astype(np.float64)
    n = M.shape[0]
    C = np.zeros((n, d+1)); C[:,1:] = M.cumsum(1)
    def spend(a,b):
        a=max(a,1); b=min(b,d)
        if a>b: return np.zeros(n)
        return C[:,b]-C[:,a-1]
    sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])
    act = (sj>0).astype(float)
    w = 0.5**np.arange(13)[:,None]
    p13 = (w*act).sum(0)/w.sum()
    p6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
    p3 = act[:3].mean(0)
    usual = (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
    expected = p13*usual
    streak = np.zeros(n)
    for j in range(13): streak += act[j]*(streak>=j)
    hazard = (act[1:]*act[:-1]).sum(0)/np.maximum(act[:-1].sum(0),1e-9)
    cv13 = sj.std(0)/(sj.mean(0)+1.0)
    Act = M>0
    gap_long = np.zeros(n); gap_n21 = np.zeros(n)
    for i in range(n):
        ad = np.where(Act[i])[0]; ad = ad[ad>=d-364]
        if len(ad)==0: continue
        gaps = np.diff(np.concatenate([[-1], ad, [d]]))-1
        gap_long[i]=gaps.max(); gap_n21[i]=(gaps>=21).sum()
    ly = spend(d-363, d-336) if d>=364 else np.zeros(n)
    ly_ratio = ly/(sj[0]+10.0)
    rvu = sj[0]/(usual+10.0)
    out = pd.DataFrame({
        'p_active13':p13,'p_active6':p6,'p_act3':p3,'usual_active':usual,'expected':expected,
        'log_expected':np.log1p(expected),'streak':streak,'hazard':hazard,'cv13':cv13,
        'gap_long':gap_long,'gap_n21':gap_n21,'ly_spend':ly,'ly_ratio':ly_ratio,'recent_vs_usual':rvu},
        index=pd.Index(hh_keys, name='household_key'))
    return out

def fn(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    if hh is None:
        tx = view.transactions
        fd = tx.groupby('household_key').day.min()
        hh = fd[fd <= d-84].index.values
    else:
        hh = hh.household_key.values if hasattr(hh,'household_key') else np.asarray(hh)
    return compute_new(view, d, hh)

F = agent_api.build_features(fn)
print('build ok', F.shape)
print(F.columns.tolist())
e1 = agent_api.load_saved('e001_history.parquet')
for d in [95, 207, 431, 543]:
    print(d, len(F[F.snapshot_day==d]), len(e1[e1.snapshot_day==d]))
agent_api.save_table(F, 'e007_new.parquet')