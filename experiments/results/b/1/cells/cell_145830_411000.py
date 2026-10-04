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