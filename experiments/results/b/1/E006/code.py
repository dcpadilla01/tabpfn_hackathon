import pandas as pd, numpy as np, agent_api
pd.set_option('display.width', 250)
t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('shape', df.shape)
print('cols:', df.columns.tolist())
y = df['future_spend_4w'].astype(float)
print('\ntarget describe:'); print(y.describe())
print('zero frac:', round(float((y==0).mean()),3))
g = df.groupby('snapshot_day')['future_spend_4w']
print('\nper-snapshot target:'); print(pd.DataFrame({'mean':g.mean().round(1),'median':g.median(),'zero':g.apply(lambda v:(v==0).mean()).round(3)}))
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
rows=[]
for c in feats:
    x = pd.to_numeric(df[c], errors='coerce')
    rows.append((c, round(x.corr(y),3), round(x.notna().mean(),3)))
cs = pd.DataFrame(rows, columns=['feat','corr_y','notna']).sort_values('corr_y', key=lambda s: s.abs(), ascending=False)
print('\nfeature corr with target:'); print(cs.to_string())
print('\nnaive single-feature MAE (train):')
res=[]
for c in feats:
    x = pd.to_numeric(df[c], errors='coerce')
    if x.notna().mean()>0.9:
        p = x.fillna(0).clip(lower=0)
        res.append((c, round(float((p-y).abs().mean()),2)))
print(sorted(res, key=lambda r: r[1])[:12])


# ---- cell ----
import pandas as pd, numpy as np, agent_api
pd.set_option('display.width',250)
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()

def feats_for(day):
    v = agent_api.snapshot(as_of_day=day)
    t = v.transactions
    hh = v.households
    g = t.groupby('household_key')
    out = pd.DataFrame(index=hh)
    for k in range(1,8):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    tmp = t.assign(age=day-t.day)
    for hl in [14,28,56]:
        w = 0.5**(tmp.age/hl)
        out[f'ew{hl}'] = (tmp.sales_value*w).groupby(tmp.household_key).sum()
    out['s84'] = t[t.day>day-84].groupby('household_key').sales_value.sum()
    out['s182'] = t[t.day>day-182].groupby('household_key').sales_value.sum()
    out['sall'] = g.sales_value.sum()
    out['dsl'] = day - g.day.max()
    out['tenure'] = day - g.day.min()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    out['gap_med'] = gaps.apply(lambda a: np.median(a) if len(a)>0 else np.nan)
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    out['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['n_gap21_182'] = g182.apply(lambda a: (a>21).sum())
    out['nb28'] = t[t.day>day-28].groupby('household_key').basket_id.nunique()
    out['nactive28'] = t[t.day>day-28].groupby('household_key').day.nunique()
    return out

frames=[]
for day in train_days:
    f = feats_for(day); f['snapshot_day']=day
    frames.append(f.reset_index().rename(columns={'index':'household_key'}))
F = pd.concat(frames, ignore_index=True).merge(tt, on=['household_key','snapshot_day'])
y = F.future_spend_4w.values.astype(float)

def mae_scale(p):
    p=np.asarray(p,dtype=float); ok=np.isfinite(p)&(p>0)
    a = np.median(y[ok]/p[ok])
    return np.abs(a*p[ok]-y[ok]).mean(), a

med3w = F[['w1','w2','w3']].median(axis=1).values
cands = {
 's28(w1)': F.w1.values,
 's84': F.s84.values,
 'mean3w': (F.w1+F.w2+F.w3).values/3,
 'med3w': med3w,
 'mean6w': F[[f'w{i}' for i in range(1,7)]].mean(axis=1).values,
 'ew14': F.ew14.values, 'ew28': F.ew28.values, 'ew56': F.ew56.values,
 'rate182': F.s182.values/182*28,
 'blend_a(0.55w1+0.25w2+0.2w3)': 0.55*F.w1.values+0.25*F.w2.values+0.20*F.w3.values,
 'blend_b(0.4w1+0.2w2+0.4med3w)': 0.4*F.w1.values+0.2*F.w2.values+0.4*med3w,
 'blend_c(0.5ew28+0.5w1)': 0.5*F.ew28.values+0.5*F.w1.values,
}
print('predictor -> scaled MAE (scale a)')
for k,v in cands.items():
    r = mae_scale(v); print(f'{k:30s} {r[0]:7.2f}  (a={r[1]:.2f})')

b = pd.cut(F.dsl,[-1,7,14,21,28,56,10000])
print('\nby days_since_last:'); print(F.groupby(b,observed=True).apply(lambda d: pd.Series({'n':len(d),'zerorate':(d.future_spend_4w==0).mean(),'y_med':d.future_spend_4w.median()})))
print('\nzero rate by n_gap21_182:'); print(F.groupby(F.n_gap21_182).apply(lambda d: pd.Series({'n':len(d),'zerorate':(d.future_spend_4w==0).mean()})))
print('\ncorr with y:'); print(F.drop(columns=['household_key','snapshot_day']).corrwith(F.future_spend_4w).sort_values(key=lambda s:s.abs(),ascending=False).round(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
pd.set_option('display.width',250)
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()

def feats_for(day):
    v = agent_api.snapshot(as_of_day=day)
    t = v.transactions
    hh = v.households
    g = t.groupby('household_key')
    out = pd.DataFrame(index=hh)
    for k in range(1,14):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    tmp = t.assign(age=day-t.day)
    for hl in [14,28,56,112]:
        w = 0.5**(tmp.age/hl)
        out[f'ew{hl}'] = (tmp.sales_value*w).groupby(tmp.household_key).sum()
    out['s84'] = t[t.day>day-84].groupby('household_key').sales_value.sum()
    out['s182'] = t[t.day>day-182].groupby('household_key').sales_value.sum()
    out['s365'] = t[t.day>day-365].groupby('household_key').sales_value.sum()
    out['dsl'] = day - g.day.max()
    out['tenure'] = day - g.day.min()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    out['gap_med'] = gaps.apply(lambda a: np.median(a) if len(a)>0 else np.nan)
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    out['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['n_gap21_182'] = g182.apply(lambda a: (a>21).sum()).astype(float)
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    out['n_gap14_84'] = g84.apply(lambda a: (a>14).sum()).astype(float)
    out['nb28'] = t[t.day>day-28].groupby('household_key').basket_id.nunique()
    out['nactive28'] = t[t.day>day-28].groupby('household_key').day.nunique()
    out['nactive84'] = t[t.day>day-84].groupby('household_key').day.nunique()
    W = out[[f'w{i}' for i in range(1,8)]]
    out['w_mean7'] = W.mean(axis=1); out['w_std7'] = W.std(axis=1); out['w_min7']=W.min(axis=1); out['w_max7']=W.max(axis=1)
    out['w_cv7'] = out.w_std7/out.w_mean7.replace(0,np.nan)
    out['spike'] = out.w1/out[[f'w{i}' for i in range(2,8)]].median(axis=1).replace(0,np.nan)
    return out

frames={}
for day in train_days:
    frames[day] = feats_for(day)
Y = {d: tt[tt.snapshot_day==d].set_index('household_key').future_spend_4w for d in train_days}

CORE = ['w1','w2','w3','w4','w5','w6','w7','ew14','ew28','ew56','ew112','s84','s182','s365','dsl','tenure','nb28','nactive28','nactive84']
NEW  = ['gap_med','gap_max182','n_gap21_182','n_gap14_84','w_std7','w_min7','w_max7','w_cv7','spike']

def design(day, cols):
    X = frames[day][cols].copy()
    return np.log1p(X.fillna(0).clip(lower=0))

def fit_ridge(Xd, yv, lam=10.0):
    mu = Xd.mean(0); sd = Xd.std(0); sd[sd==0]=1
    Z = (Xd-mu)/sd
    A = Z.T@Z + lam*np.eye(Z.shape[1]); b = Z.T@yv
    return (mu,sd,np.linalg.solve(A,b))

def pred_ridge(m, Xd):
    mu,sd,coef = m
    Z = (Xd-mu)/sd
    return np.expm1(np.clip(Z@coef,0,20))

def eval_sets(cols, stacked=False):
    preds={}; actuals={}
    for D in [403,431]:
        prior = [s for s in train_days if s+28<=D]
        Xtr = pd.concat([design(s,cols) for s in prior])
        ytr = np.log1p(pd.concat([Y[s] for s in prior]).reindex(Xtr.index).fillna(0).values)
        m = fit_ridge(Xtr.values, ytr)
        Xd = design(D,cols)
        p = pred_ridge(m, Xd.values)
        yv = Y[D].reindex(frames[D].index).fillna(0).values
        preds[D]=p; actuals[D]=yv
    p = np.concatenate([preds[403],preds[431]]); yv=np.concatenate([actuals[403],actuals[431]])
    p = np.clip(p,0,None)
    return np.abs(p-yv).mean()

print('local ridge proxy, eval on days 403+431 (MAE):')
print('  core (E001-like numeric)      :', round(eval_sets(CORE),2))
print('  core + gap/volatility         :', round(eval_sets(CORE+NEW),2))
print('  core+new, log windows only    :', round(eval_sets(['log_'+c for c in []] or CORE[:6]+NEW),2))
# also: simple scaled baseline on same eval set
p = np.concatenate([(0.9*frames[403].w1).values,(0.9*frames[431].w1).values])
yv = np.concatenate([Y[403].reindex(frames[403].index).fillna(0).values, Y[431].reindex(frames[431].index).fillna(0).values])
print('  scaled w1 baseline            :', round(np.abs(p-yv).mean(),2))
# which NEW features matter: per-feature corr with residual of core model
Xtr = pd.concat([design(s,CORE) for s in train_days if s+28<=431])
ytr = np.log1p(pd.concat([Y[s] for s in train_days if s+28<=431]).reindex(Xtr.index).fillna(0).values)
m = fit_ridge(Xtr.values,ytr)
res = ytr - np.clip((Xtr.values-m[0])/m[1]@m[2],0,20)
Xn = pd.concat([design(s,NEW) for s in train_days if s+28<=431])
print('\ncorr(new feat, ridge residual):')
print(Xn.corrwith(pd.Series(res,index=Xn.index)).sort_values(key=lambda s:s.abs(),ascending=False).round(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
pd.set_option('display.width',250)
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()

def feats_for(day):
    v = agent_api.snapshot(as_of_day=day)
    t = v.transactions
    hh = v.households
    g = t.groupby('household_key')
    out = pd.DataFrame(index=hh)
    for k in range(1,14):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    tmp = t.assign(age=day-t.day)
    for hl in [14,28,56,112]:
        w = 0.5**(tmp.age/hl)
        out[f'ew{hl}'] = (tmp.sales_value*w).groupby(tmp.household_key).sum()
    out['s84'] = t[t.day>day-84].groupby('household_key').sales_value.sum()
    out['s182'] = t[t.day>day-182].groupby('household_key').sales_value.sum()
    out['s365'] = t[t.day>day-365].groupby('household_key').sales_value.sum()
    out['dsl'] = day - g.day.max()
    out['tenure'] = day - g.day.min()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    out['gap_med'] = gaps.apply(lambda a: np.median(a) if len(a)>0 else np.nan)
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    out['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['n_gap21_182'] = g182.apply(lambda a: (a>21).sum()).astype(float)
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    out['n_gap14_84'] = g84.apply(lambda a: (a>14).sum()).astype(float)
    out['nb28'] = t[t.day>day-28].groupby('household_key').basket_id.nunique()
    out['nactive28'] = t[t.day>day-28].groupby('household_key').day.nunique()
    out['nactive84'] = t[t.day>day-84].groupby('household_key').day.nunique()
    W = out[[f'w{i}' for i in range(1,8)]]
    out['w_mean7'] = W.mean(axis=1); out['w_std7'] = W.std(axis=1); out['w_min7']=W.min(axis=1); out['w_max7']=W.max(axis=1)
    out['w_cv7'] = out.w_std7/out.w_mean7.replace(0,np.nan)
    out['spike'] = out.w1/out[[f'w{i}' for i in range(2,8)]].median(axis=1).replace(0,np.nan)
    out = out.reset_index().rename(columns={'index':'household_key'})
    out['key'] = out.snapshot_day if False else out['household_key'].astype(str)+'_'+str(day)
    return out

frames={}
for day in train_days:
    frames[day] = feats_for(day)
Y = {d: tt[tt.snapshot_day==d].set_index('household_key').future_spend_4w for d in train_days}
for d in train_days:
    frames[d]['y'] = frames[d]['household_key'].map(Y[d])

CORE = ['w1','w2','w3','w4','w5','w6','w7','ew14','ew28','ew56','ew112','s84','s182','s365','dsl','tenure','nb28','nactive28','nactive84']
NEW  = ['gap_med','gap_max182','n_gap21_182','n_gap14_84','w_std7','w_min7','w_max7','w_cv7','spike']

def design(df, cols):
    return np.log1p(df[cols].fillna(0).clip(lower=0)).values

def fit_ridge(Xd, yv, lam=10.0):
    mu = Xd.mean(0); sd = Xd.std(0); sd[sd==0]=1
    Z = (Xd-mu)/sd
    A = Z.T@Z + lam*np.eye(Z.shape[1]); b = Z.T@yv
    return (mu,sd,np.linalg.solve(A,b))

def eval_sets(cols):
    preds=[]; actuals=[]
    for D in [403,431]:
        prior = [s for s in train_days if s+28<=D]
        Xtr = np.vstack([design(frames[s],cols) for s in prior])
        ytr = np.log1p(np.concatenate([frames[s]['y'].fillna(0).values for s in prior]))
        m = fit_ridge(Xtr, ytr)
        Xd = design(frames[D],cols)
        p = np.expm1(np.clip((Xd-m[0])/m[1]@m[2],0,20))
        preds.append(p); actuals.append(frames[D]['y'].fillna(0).values)
    p=np.clip(np.concatenate(preds),0,None); yv=np.concatenate(actuals)
    return np.abs(p-yv).mean()

print('local ridge proxy, eval days 403+431 (MAE):')
print('  core (E001-like)      :', round(eval_sets(CORE),2))
print('  core + gap/volatility :', round(eval_sets(CORE+NEW),2))
p = np.concatenate([(0.9*frames[403].w1).values,(0.9*frames[431].w1).values])
yv = np.concatenate([frames[403]['y'].fillna(0).values, frames[431]['y'].fillna(0).values])
print('  scaled w1 baseline    :', round(np.abs(p-yv).mean(),2))
Xtr = np.vstack([design(frames[s],CORE) for s in train_days if s+28<=431])
ytr = np.log1p(np.concatenate([frames[s]['y'].fillna(0).values for s in train_days if s+28<=431]))
m = fit_ridge(Xtr,ytr)
res = ytr - np.clip((Xtr-m[0])/m[1]@m[2],0,20)
Xn = np.vstack([design(frames[s],NEW) for s in train_days if s+28<=431])
Xn_df = pd.DataFrame(Xn, columns=NEW)
print('\ncorr(new feat, ridge residual):')
print(Xn_df.corrwith(pd.Series(res)).sort_values(key=lambda s:s.abs(),ascending=False).round(3))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
# rebuild minimal frames (w1..w7, ew28, s84, dsl) to debug the ridge proxy
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()
def feats_for(day):
    v = agent_api.snapshot(as_of_day=day); t = v.transactions
    out = pd.DataFrame(index=v.households)
    for k in range(1,8):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    out['dsl'] = day - t.groupby('household_key').day.max()
    out = out.reset_index().rename(columns={'index':'household_key'})
    out['y'] = out['household_key'].map(tt[tt.snapshot_day==day].set_index('household_key').future_spend_4w)
    return out
frames = {d: feats_for(d) for d in train_days}
COLS=['w1','w2','w3','w4','w5','w6','w7','dsl']
def design(df): return np.log1p(df[COLS].fillna(0).clip(lower=0)).values
for D in [403,431]:
    prior=[s for s in train_days if s+28<=D]
    Xtr=np.vstack([design(frames[s]) for s in prior])
    ytr=np.log1p(np.concatenate([frames[s]['y'].fillna(0).values for s in prior]))
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Z=(Xtr-mu)/sd
    lam=10.0
    coef=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@ytr)
    Xd=design(frames[D]); Zd=(Xd-mu)/sd
    raw=Zd@coef
    p=np.expm1(np.clip(raw,0,20)); yv=frames[D]['y'].fillna(0).values
    print(f'day {D}: raw pred quantiles', np.round(np.quantile(raw,[0,.5,.9,.99,1]),2))
    print('   p quantiles', np.round(np.quantile(p,[0,.5,.9,.99,1]),1), ' y quantiles', np.round(np.quantile(yv,[0,.5,.9,.99,1]),1))
    print('   MAE no-clip-low:', round(np.abs(np.expm1(np.clip(raw,None,20))-yv).mean(),2), ' MAE clip[0,20]:', round(np.abs(p-yv).mean(),2))
    print('   frac raw<0:', round(float((raw<0).mean()),3))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()
def feats_for(day):
    v = agent_api.snapshot(as_of_day=day); t = v.transactions
    out = pd.DataFrame(index=v.households)
    for k in range(1,8):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    out['dsl'] = day - t.groupby('household_key').day.max()
    out = out.reset_index().rename(columns={'index':'household_key'})
    out['y'] = out['household_key'].map(tt[tt.snapshot_day==day].set_index('household_key').future_spend_4w)
    return out
frames = {d: feats_for(d) for d in train_days}
COLS=['w1','w2','w3','w4','w5','w6','w7','dsl']
def design(df): return np.log1p(df[COLS].fillna(0).clip(lower=0)).values
s=403
prior=[x for x in train_days if x+28<=s]
Xtr=np.vstack([design(frames[x]) for x in prior])
ytr=np.log1p(np.concatenate([frames[x]['y'].fillna(0).values for x in prior]))
print('n rows', Xtr.shape, 'y log quantiles', np.round(np.quantile(ytr,[0,.25,.5,.75,1]),2))
print('X col means', np.round(Xtr.mean(0),2))
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
Z=(Xtr-mu)/sd
print('Z diag', np.round(np.diag(Z.T@Z),0))
coef=np.linalg.solve(Z.T@Z+1e-6*np.eye(Z.shape[1]), Z.T@ytr)
print('coef', np.round(coef,3))
tr_pred=Z@coef
print('train log-space MAE', round(np.abs(tr_pred-ytr).mean(),3), 'train pred q', np.round(np.quantile(tr_pred,[0,.5,1]),2))
# in-sample is fine? then check day-403 design distribution vs train
Xd=design(frames[403]); Zd=(Xd-mu)/sd
print('day403 Zd mean', np.round(Zd.mean(0),2), 'train Z mean', np.round(Z.mean(0),2))
print('day403 raw pred q', np.round(np.quantile(Zd@coef,[0,.5,.9,1]),2))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
day=403
v = agent_api.snapshot(as_of_day=day); t = v.transactions
out = pd.DataFrame(index=v.households)
for k in range(1,8):
    lo = day-28*k
    out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
out = out.reset_index()
print('reset_index cols:', out.columns.tolist(), 'index name was:', v.households.name if hasattr(v.households,'name') else type(v.households))
print(out.head(3))
out['y'] = out['household_key'].map(tt[tt.snapshot_day==day].set_index('household_key').future_spend_4w)
print('y notna frac:', round(out.y.notna().mean(),3), 'n rows', len(out))
L = np.log1p(out[['w1','w2','w3','w4','w5','w6','w7']].fillna(0).clip(lower=0))
Ly = np.log1p(out.y.fillna(0))
print('corr(log w, log y):'); print(L.corrwith(Ly).round(3))
print('corr raw w1 vs y:', round(out.w1.fillna(0).corr(out.y.fillna(0)),3))
# check tt day 403 coverage
print('tt day403 n:', (tt.snapshot_day==403).sum())

# ---- cell ----
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
print('tt shape:', tt.shape)
print('tt per-day counts:'); print(tt.groupby('snapshot_day').size())
t = agent_api.load_saved('e001_history.parquet')
print('\nE001 rows per snapshot_day:'); print(t.groupby('snapshot_day').size())
# what is v.households at various days?
for d in [95, 207, 403, 459]:
    v = agent_api.snapshot(as_of_day=d)
    print(f'\nas_of_day={d}: v.households n={len(v.households)}')
    # households with first purchase <= d-84 among transactions up to d
    tr = v.transactions
    fp = tr.groupby('household_key').day.min()
    n84 = int((fp <= d-84).sum())
    print('  households with first purchase <= d-84 (in txns up to d):', n84)
    print('  households in txns up to d:', tr.household_key.nunique())
    print('  tt count at d (if train):', int((tt.snapshot_day==d).sum()))
    hv = set(v.households); tv = set(tt[tt.snapshot_day==d].household_key)
    print('  v.households subset of tt?', hv <= tv, ' tt - v:', len(tv-hv), ' v - tt:', len(hv-tv))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
v459 = agent_api.snapshot()  # max view
t459 = v459.transactions
fp_full = t459.groupby('household_key').day.min()   # first purchase over all visible data
print('households with any txn <=459:', len(fp_full))
for d in [95, 123, 207, 403, 431, 459]:
    rows = set(tt[tt.snapshot_day==d].household_key)
    elig_full = set(fp_full[fp_full <= d-84].index)          # full-data first purchase
    v = agent_api.snapshot(as_of_day=d)
    tv = v.transactions
    elig_trunc = set(tv.groupby('household_key').day.min().pipe(lambda s: s[s<=d-84]).index)  # truncated
    print(f'day {d}: rows={len(rows)} elig_full={len(elig_full)} elig_trunc={len(elig_trunc)} '
          f'rows==elig_full:{rows==elig_full} rows==elig_trunc:{rows==elig_trunc} '
          f'rows_no_txn_yet={len(rows - set(tv.household_key.unique()))}')
print()
print('type of v.households at 403:', type(agent_api.snapshot(as_of_day=403).households))
print('type at 95:', type(agent_api.snapshot(as_of_day=95).households))
print('type at 459:', type(v459.households))
hh403 = agent_api.snapshot(as_of_day=403).households
if hh403 is not None:
    print('n at 403:', len(hh403), '== rows?', set(hh403)==set(tt[tt.snapshot_day==403].household_key))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()

def feats_for(day):
    v = agent_api.snapshot(as_of_day=day); t = v.transactions
    fp = t.groupby('household_key').day.min()
    idx = fp[fp <= day-84].index                      # exact row set for this snapshot
    out = pd.DataFrame(index=idx)
    for k in range(1,14):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    tmp = t.assign(age=day-t.day)
    for hl in [14,28,56,112]:
        w = 0.5**(tmp.age/hl)
        out[f'ew{hl}'] = (tmp.sales_value*w).groupby(tmp.household_key).sum()
    out['s84'] = t[t.day>day-84].groupby('household_key').sales_value.sum()
    out['s182'] = t[t.day>day-182].groupby('household_key').sales_value.sum()
    out['dsl'] = day - t.groupby('household_key').day.max()
    out['tenure'] = day - fp
    out['nb28'] = t[t.day>day-28].groupby('household_key').basket_id.nunique()
    out['nactive28'] = t[t.day>day-28].groupby('household_key').day.nunique()
    out['nactive84'] = t[t.day>day-84].groupby('household_key').day.nunique()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    gaps = ud.reindex(idx).apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    out['n_gap21_182'] = g182.apply(lambda a: (a>21).sum()).astype(float)
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    out['n_gap14_84'] = g84.apply(lambda a: (a>14).sum()).astype(float)
    out['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else 0).astype(float)
    W = out[[f'w{i}' for i in range(1,8)]]
    out['w_std7'] = W.std(axis=1); out['w_max7'] = W.max(axis=1); out['w_min7'] = W.min(axis=1)
    out['w_cv7'] = out.w_std7/out.w_mean7 if False else out.w_std7/W.mean(axis=1).replace(0,np.nan)
    out['spike'] = out.w1/out[[f'w{i}' for i in range(2,8)]].median(axis=1).replace(0,np.nan)
    out['y'] = out.index.map(tt[tt.snapshot_day==day].set_index('household_key').future_spend_4w)
    return out

frames = {d: feats_for(d) for d in train_days}
for d in [403,431]: print(d, 'rows', len(frames[d]), 'y notna', frames[d].y.notna().sum())

CORE = ['w1','w2','w3','w4','w5','w6','w7','ew14','ew28','ew56','ew112','s84','s182','dsl','tenure','nb28','nactive28','nactive84']
NEW  = ['w8','w9','w10','w11','w12','w13','gap_max182','n_gap21_182','n_gap14_84','w_std7','w_max7','w_min7','w_cv7','spike']

def design(df, cols): return np.log1p(df[cols].fillna(0).clip(lower=0)).values

def eval_sets(cols, lam=1.0):
    preds=[]; actuals=[]
    for D in [403,431]:
        prior=[s for s in train_days if s+28<=D]
        Xtr=np.vstack([design(frames[s],cols) for s in prior])
        ytr=np.log1p(np.concatenate([frames[s]['y'].fillna(0).values for s in prior]))
        mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
        Z=(Xtr-mu)/sd
        Zc=np.hstack([Z,np.ones((len(Z),1))])
        coef=np.linalg.solve(Zc.T@Zc+lam*np.eye(Zc.shape[1]), Zc.T@ytr)
        Xd=design(frames[D],cols); Zd=(Xd-mu)/sd
        raw=np.hstack([Zd,np.ones((len(Zd),1))])@coef
        preds.append(np.clip(np.expm1(raw),0,None)); actuals.append(frames[D]['y'].fillna(0).values)
    return np.abs(np.concatenate(preds)-np.concatenate(actuals)).mean()

print('\nlocal ridge proxy (with intercept), eval 403+431:')
print('  core          :', round(eval_sets(CORE),2))
print('  core+new      :', round(eval_sets(CORE+NEW),2))
print('  core+new lam3 :', round(eval_sets(CORE+NEW,3.0),2))
print('  core lam3     :', round(eval_sets(CORE,3.0),2))
p=np.concatenate([(0.9*frames[403].w1).values,(0.9*frames[431].w1).values])
yv=np.concatenate([frames[403].y.fillna(0).values,frames[431].y.fillna(0).values])
print('  scaled w1     :', round(np.abs(p-yv).mean(),2))

# ---- cell ----
import pandas as pd, numpy as np, agent_api

def make_feats(view, day):
    t = view.transactions
    hh = pd.Index(view.households)
    g = t.groupby('household_key')
    # ---------- NEW: temporal-shape features ----------
    nf = pd.DataFrame(index=hh)
    for k in range(1,14):
        lo = day-28*k
        nf[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    age = day - t.day
    for hl in [28,56]:
        nf[f'ew{hl}'] = (t.sales_value*(0.5**(age/hl))).groupby(t.household_key).sum()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique())).reindex(hh)
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    nf['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else 0.0).astype(float)
    nf['n_gap21_182'] = g182.apply(lambda a: float((a>21).sum()))
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    nf['n_gap14_84'] = g84.apply(lambda a: float((a>14).sum()))
    nf['gap_med'] = gaps.apply(lambda a: float(np.median(a)) if len(a)>0 else np.nan)
    W = nf[[f'w{i}' for i in range(1,8)]]
    Wm = W.mean(axis=1); Ws = W.std(axis=1)
    nf['w_std7'] = Ws; nf['w_max7'] = W.max(axis=1); nf['w_min7'] = W.min(axis=1)
    nf['w_cv7'] = np.where(Wm>0, Ws/Wm.replace(0,np.nan), 0.0)
    med = W[[f'w{i}' for i in range(2,8)]].median(axis=1)
    nf['spike'] = np.where(med>0, nf['w1']/med.replace(0,np.nan), np.where(nf['w1']>0, 10.0, 0.0))
    for c in [f'w{k}' for k in range(1,14)]+['ew28','ew56','gap_max182','n_gap21_182','n_gap14_84']:
        nf[c] = nf[c].fillna(0.0)
    # ---------- BASE: E001 ----------
    try:
        base = agent_api.load_saved('e001_history.parquet')
        if 'future_spend_4w' in base.columns: base = base.drop(columns=['future_spend_4w'])
        b = base[base['snapshot_day']==day].drop(columns=['snapshot_day']).set_index('household_key')
        res = b.join(nf, how='left')
    except Exception as e:
        print('load_saved inside fn failed:', e)
        d = t.day
        ef = pd.DataFrame(index=hh)
        ef['spend_7']  = t[d>day-7].groupby('household_key').sales_value.sum()
        ef['spend_28'] = nf['w1']
        ef['spend_56'] = t[d>day-56].groupby('household_key').sales_value.sum()
        ef['spend_84'] = t[d>day-84].groupby('household_key').sales_value.sum()
        ef['spend_182']= t[d>day-182].groupby('household_key').sales_value.sum()
        ef['spend_365']= t[d>day-365].groupby('household_key').sales_value.sum()
        ef['spend_all']= g.sales_value.sum()
        ef['nb_28'] = t[d>day-28].groupby('household_key').basket_id.nunique()
        ef['nb_56'] = t[d>day-56].groupby('household_key').basket_id.nunique()
        ef['nb_all'] = g.basket_id.nunique()
        ef['avg_basket_28'] = ef['spend_28']/ef['nb_28'].replace(0,np.nan)
        ef['days_since_last'] = day - g.day.max()
        ef['spend_prev28'] = nf['w2']
        ef['trend28'] = ef['spend_28'] - ef['spend_prev28']
        ef['qty_28'] = t[d>day-28].groupby('household_key').quantity.sum()
        ef['nprod_28'] = t[d>day-28].groupby('household_key').product_id.nunique()
        ef['spend_ly28'] = t[(d>day-364)&(d<=day-336)].groupby('household_key').sales_value.sum()
        fp = g.day.min()
        ef['ly_avail'] = (fp<=day-364).astype(float)
        ef['log_spend_28'] = np.log1p(ef['spend_28'].fillna(0))
        ef['log_spend_all'] = np.log1p(ef['spend_all'].fillna(0))
        ef['weekly_rate_84'] = ef['spend_84'].fillna(0)/84*7
        ef = ef.fillna({'spend_7':0,'spend_56':0,'spend_84':0,'spend_182':0,'spend_365':0,'spend_all':0,
                        'nb_28':0,'nb_56':0,'nb_all':0,'days_since_last':999,'spend_prev28':0,'trend28':0,
                        'qty_28':0,'nprod_28':0,'spend_ly28':0,'ly_avail':0})
        demo = view.demographics if hasattr(view,'demographics') else None
        ef['has_demographics'] = ef.index.isin(set(demo.household_key)).astype(float) if demo is not None else 0.0
        if demo is not None: ef = ef.join(demo.set_index('household_key').drop(columns=['household_key']), how='left')
        ef['snapshot_day_index'] = float(day)
        ef['week_of_year'] = float(((day+8)//7) % 52)
        res = ef.join(nf, how='left')
    return res

T = agent_api.build_features(make_feats)
print('built shape:', T.shape)
print('cols:', T.columns.tolist())
print('rows per snapshot:'); print(T.groupby('snapshot_day').size())
print('any all-NaN new cols:', [c for c in ['w8','w13','ew28','gap_max182','spike','w_cv7'] if T[c].notna().sum()==0])
path = agent_api.save_table(T, 'e006_seq_gaps')
print('saved:', path)

# ---- cell ----
import pandas as pd, numpy as np, agent_api

def make_feats(view, day):
    t = view.transactions
    hh = pd.Index(view.households)
    g = t.groupby('household_key')
    d = t.day
    nf = pd.DataFrame(index=hh)
    for k in range(1,14):
        lo = day-28*k
        nf[f'w{k}'] = t[(d>lo)&(d<=lo+28)].groupby('household_key').sales_value.sum()
    age = day - d
    for hl in [28,56]:
        nf[f'ew{hl}'] = (t.sales_value*(0.5**(age/hl))).groupby(t.household_key).sum()
    ud = g.day.apply(lambda s: np.sort(s.unique())).reindex(hh)
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    nf['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else 0.0).astype(float)
    nf['n_gap21_182'] = g182.apply(lambda a: float((a>21).sum()))
    g84 = gaps.apply(lambda a: a[a<=84] if len(a)>0 else np.array([]))
    nf['n_gap14_84'] = g84.apply(lambda a: float((a>14).sum()))
    nf['gap_med'] = gaps.apply(lambda a: float(np.median(a)) if len(a)>0 else np.nan)
    W = nf[[f'w{i}' for i in range(1,8)]]
    Wm = W.mean(axis=1); Ws = W.std(axis=1)
    nf['w_std7'] = Ws; nf['w_max7'] = W.max(axis=1); nf['w_min7'] = W.min(axis=1)
    nf['w_cv7'] = np.where(Wm>0, Ws/Wm.replace(0,np.nan), 0.0)
    med = W[[f'w{i}' for i in range(2,8)]].median(axis=1)
    nf['spike'] = np.where(med>0, nf['w1']/med.replace(0,np.nan), np.where(nf['w1']>0, 10.0, 0.0))
    ef = pd.DataFrame(index=hh)
    ef['spend_7']  = t[d>day-7].groupby('household_key').sales_value.sum()
    ef['spend_28'] = nf['w1']
    ef['spend_56'] = t[d>day-56].groupby('household_key').sales_value.sum()
    ef['spend_84'] = t[d>day-84].groupby('household_key').sales_value.sum()
    ef['spend_182']= t[d>day-182].groupby('household_key').sales_value.sum()
    ef['spend_365']= t[d>day-365].groupby('household_key').sales_value.sum()
    ef['spend_all']= g.sales_value.sum()
    ef['nb_28'] = t[d>day-28].groupby('household_key').basket_id.nunique()
    ef['nb_56'] = t[d>day-56].groupby('household_key').basket_id.nunique()
    ef['nb_all'] = g.basket_id.nunique()
    ef['avg_basket_28'] = ef['spend_28']/ef['nb_28'].replace(0,np.nan)
    ef['days_since_last'] = day - g.day.max()
    ef['spend_prev28'] = nf['w2']
    ef['trend28'] = ef['spend_28'] - ef['spend_prev28']
    ef['qty_28'] = t[d>day-28].groupby('household_key').quantity.sum()
    ef['nprod_28'] = t[d>day-28].groupby('household_key').product_id.nunique()
    ef['spend_ly28'] = t[(d>day-364)&(d<=day-336)].groupby('household_key').sales_value.sum()
    fp = g.day.min()
    ef['ly_avail'] = (fp<=day-364).astype(float)
    ef['log_spend_28'] = np.log1p(ef['spend_28'].fillna(0))
    ef['log_spend_all'] = np.log1p(ef['spend_all'].fillna(0))
    ef['weekly_rate_84'] = ef['spend_84'].fillna(0)/84*7
    ef['index'] = ef.index.astype(float)
    zero_fill = ['spend_7','spend_56','spend_84','spend_182','spend_365','spend_all','nb_28','nb_56','nb_all',
                 'spend_prev28','trend28','qty_28','nprod_28','spend_ly28','ly_avail']
    ef[zero_fill] = ef[zero_fill].fillna(0.0)
    ef['days_since_last'] = ef['days_since_last'].fillna(999.0)
    demo = view.demographics
    dm = demo.set_index('household_key') if 'household_key' in demo.columns else demo
    ef['has_demographics'] = ef.index.isin(dm.index).astype(float)
    ef = ef.join(dm, how='left')
    ef['snapshot_day_index'] = float(day)
    ef['week_of_year'] = float(((day+8)//7) % 52)
    return ef.join(nf, how='left')

T = agent_api.build_features(make_feats)
print('built shape:', T.shape)
print('rows per snapshot:', T.groupby('snapshot_day').size().to_dict())
print('NaN-only new cols:', [c for c in ['w8','w13','ew28','gap_max182','spike','w_cv7','gap_med'] if T[c].notna().sum()==0])
path = agent_api.save_table(T, 'e006_seq_gaps')
print('saved:', path)