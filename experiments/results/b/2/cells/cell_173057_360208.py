import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
sub = m[m.snapshot_day<347][feat]
nun = sub.nunique(); const = nun[nun<=1].index.tolist()
corr = sub.corr().abs().fillna(0)
drop=set(const)
for i,a in enumerate(feat):
    if a in drop: continue
    for b in feat[i+1:]:
        if b in drop: continue
        if corr.loc[a,b]>0.999: drop.add(b)
feat2=[c for c in feat if c not in drop]

def prep(tr, te, cols, alpha):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Ztr=(Xtr-mu)/sd; Zte=(Xte-mu)/sd
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    ytr=tr.future_spend_4w.to_numpy(float); p=Ztr.shape[1]
    A=Ztr.T@Ztr+alpha*np.eye(p); A[-1,-1]-=alpha
    w=np.linalg.pinv(A)@(Ztr.T@ytr)
    return Zte@w
def cv(mm, cols, alphas, eval_days=(347,375,403,431)):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=mm[mm.snapshot_day<D]; te=mm[mm.snapshot_day==D]
            per.append(np.mean(np.abs(prep(tr,te,cols,a)-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

X = m[feat2].to_numpy(float)
med = np.nanmedian(X, axis=0)
Xf = np.where(np.isnan(X), med, X)
sk = np.abs(pd.DataFrame(Xf, columns=feat2).skew().to_numpy())
heavy = [feat2[i] for i in range(len(feat2)) if sk[i]>2]
print('heavy count', len(heavy))

# A: add log1p copies of heavy features
A = m[['household_key','snapshot_day','future_spend_4w']].copy()
for c in feat2: A[c]=m[c].to_numpy(float)
for c in heavy: A['lg_'+c]=np.log1p(np.maximum(m[c].to_numpy(float),0))
print('A +log copies:', cv(A, feat2+['lg_'+c for c in heavy], [3000,10000]))

# B: interactions
B = A.copy()
for c in ['spend_84','fwd28_mean','wk_avg_84','spend_28']:
    B['ix_'+c+'_slope']=m[c].to_numpy(float)*m['wk_slope'].fillna(0).to_numpy(float)
    B['ix_'+c+'_burst']=m[c].to_numpy(float)*m['c_burst7'].fillna(0).to_numpy(float)
    B['ix_'+c+'_size']=m[c].to_numpy(float)*m['size_ord'].fillna(0).to_numpy(float)
    B['ix_'+c+'_rec']=m[c].to_numpy(float)*m['z_rec45'].fillna(0).to_numpy(float)
inter=[c for c in B.columns if c.startswith('ix_')]
print('B +interactions:', cv(B, feat2+['lg_'+c for c in heavy]+inter, [3000,10000]))

# C: sqrt copies of heavy
C = B.copy()
for c in heavy: C['sq_'+c]=np.sqrt(np.maximum(m[c].to_numpy(float),0))
print('C +sqrt copies:', cv(C, feat2+['lg_'+c for c in heavy]+inter+['sq_'+c for c in heavy], [3000,10000]))