import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Z=np.clip((Xtr-mu)/sd,-30,30); Zv=np.clip((Xva-mu)/sd,-30,30)
    Z=np.c_[np.ones(len(Z)),Z]; Zv=np.c_[np.ones(len(Zv)),Zv]
    A=Z.T@Z+alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return np.clip(Zv@np.linalg.solve(A,Z.T@ytr),0,None)

hist = agent_api.snapshot(431).transactions[['household_key','day','sales_value','basket_id']]
hh_needed = agent_api.snapshot(431).households
print('hh at 431:', len(hh_needed), 'trans rows:', len(hist))
g = hist.groupby(['household_key','day'], as_index=False).sales_value.sum()
# spend in (a,b] helper via per-household sorted day sums
hh = g.household_key.values; dy = g.day.values; sv = g.sales_value.values
order = np.lexsort((dy, hh)); hh,dy,sv = hh[order],dy[order],sv[order]
starts = np.searchsorted(hh, np.unique(hh), side='left'); ends = np.searchsorted(hh, np.unique(hh), side='right')
uniq = np.unique(hh); hidx = {h:i for i,h in enumerate(uniq)}
def win(h, lo, hi):  # spend in (lo, hi]
    i = hidx[h]
    d, s = dy[starts[i]:ends[i]], sv[starts[i]:ends[i]]
    m = (d>lo)&(d<=hi)
    return s[m].sum()
def vec(h, t):
    s28=win(h,t-28,t); s84=win(h,t-84,t); s364=win(h,t-364,t)
    k1=s28; k2=win(h,t-56,t-28); k3=win(h,t-84,t-56); k4=win(h,t-112,t-84)
    ew=(k1*8+k2*4+k3*2+k4)/15.
    ya=win(h,t-392,t-364)
    return [np.log1p(s28),np.log1p(s84),np.log1p(s364),np.log1p(k1),np.log1p(k2),np.log1p(k3),np.log1p(k4),
            np.log1p(ew), np.log1p(ya), np.log1p(s84/4.)]

# pooled panel at as-of day 431: pairs (h, t) t in 84..403 step 28
rows=[]
for t in range(84, 404, 28):
    for h in uniq:
        x = vec(h,t); y = win(h,t,t+28)
        rows.append(x+[y])
P = np.array(rows)
print('pooled panel:', P.shape)
Xp, yp = P[:,:-1], P[:,-1]
# evaluate as feature: fit pooled ridge at 431 for all needed hh, then inner eval
base = agent_api.load_saved('e009_demo.parquet')
tt = agent_api.train_targets()
df = base.merge(tt, on=['household_key','snapshot_day'], how='left')
tr_days = agent_api.snapshot_days()['train']
fcols=[c for c in df.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
for inner_day in [403,431]:
    # pooled model as-of inner_day
    rows=[]
    for t in range(84, inner_day-27, 28):
        for h in uniq:
            rows.append(vec(h,t)+[win(h,t,t+28)])
    P=np.array(rows); Xp,yp=P[:,:-1],P[:,-1]
    mu=Xp.mean(0); sd=Xp.std(0); sd[sd==0]=1
    Z=np.c_[np.ones(len(Xp)),np.clip((Xp-mu)/sd,-30,30)]
    A=Z.T@Z+50*np.eye(Z.shape[1]); A[-1,-1]-=50
    w=np.linalg.solve(A,Z.T@yp)
    # predict for households at inner_day
    pred={h: float(np.clip(np.r_[1,np.clip((np.array(vec(h,inner_day))-mu)/sd,-30,30)]@w,0,None)) for h in uniq}
    df['pooled']=df.household_key.map(pred)
    trin=df[np.isin(df.snapshot_day.values,[x for x in tr_days if x<inner_day])]
    inner=df[df.snapshot_day.values==inner_day]
    Xtr=np.c_[trin[fcols].apply(pd.to_numeric,errors='coerce').fillna(0).values, trin[['pooled']].values]
    Xin=np.c_[inner[fcols].apply(pd.to_numeric,errors='coerce').fillna(0).values, inner[['pooled']].values]
    p=ridge_fit_pred(Xtr,trin.future_spend_4w.values,Xin,3000.)
    mae=np.abs(p-inner.future_spend_4w.values).mean()
    corr=np.corrcoef(inner.pooled, inner.future_spend_4w)[0,1]
    print(f'd{inner_day}: innerMAE={mae:.3f} pooled-corr={corr:.3f}')
