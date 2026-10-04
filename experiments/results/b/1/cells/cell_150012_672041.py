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