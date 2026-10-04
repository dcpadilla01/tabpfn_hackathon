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
