import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
print(t.shape, t.columns[:5].tolist(), t.columns[-5:].tolist())
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
key = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in key]
num = df[feats].select_dtypes(include=[np.number]).columns.tolist()
nonnum = [c for c in feats if c not in num]
print('n feats', len(feats), 'numeric', len(num), 'nonnum', nonnum)
print('zero-target share', (df.future_spend_4w==0).mean())
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
print('tr', len(tr), 'iv', len(iv))

def ev(cols, mode='raw', alpha=10.0, topk=None):
    A = tr[cols].astype(float).copy(); B = iv[cols].astype(float).copy()
    if mode=='log1p':
        for c in cols:
            if A[c].min() >= 0 and A[c].max() > 10:
                A[c] = np.log1p(A[c]); B[c] = np.log1p(B[c])
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    if topk:
        w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        idx=np.argsort(-np.abs(w))[:topk]
        return ev([cols[i] for i in idx], mode, alpha)
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

for a in [1.0,10.0,100.0,300.0]:
    r=ev(num,alpha=a); print('alpha %g: train %.3f inner-val %.3f'%(a,r[0],r[1]))
print('log1p a10:', ev(num,mode='log1p',alpha=10.0))
print('top100 a10:', ev(num,alpha=10.0,topk=100))
print('top200 a10:', ev(num,alpha=10.0,topk=200))
cm = df[num].corrwith(df.future_spend_4w)
print(cm.abs().sort_values(ascending=False).head(25))
print(df.groupby('snapshot_day').future_spend_4w.agg(['mean','count']))
