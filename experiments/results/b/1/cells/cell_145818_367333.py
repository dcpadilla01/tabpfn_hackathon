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