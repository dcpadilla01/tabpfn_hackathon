import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e013_denoise.parquet')
print('shape', df.shape)
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(feat))
for i in range(0, len(feat), 8):
    print(' | '.join(feat[i:i+8]))
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
y = m['future_spend_4w']
print('y stats', y.describe())
print('zero share', round((y==0).mean(),3))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(mean='mean', med='median', zero=lambda s:(s==0).mean()))
num = m[feat].select_dtypes(include=[np.number])
corr = num.corrwith(y)
print('TOP |corr|:')
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(30).round(3))
sk = num.skew()
print('MOST SKEWED:'); print(sk.sort_values(ascending=False).head(25).round(1))
print('n |skew|>3:', int((sk.abs()>3).sum()))

# ---- cell ----
import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('rows', len(m), 'feat', len(feat))

def fit_predict(tr, te, cols, alpha, clip_q=None, logskew=False):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if clip_q is not None:
        lo = np.quantile(Xtr, clip_q, axis=0); hi = np.quantile(Xtr, 1-clip_q, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
    if logskew:
        sk = np.abs(pd.DataFrame(Xtr).skew().to_numpy())
        ms = sk > 2
        Xtr = np.where(ms[None,:], np.sign(Xtr)*np.log1p(np.abs(Xtr)), Xtr)
        Xte = np.where(ms[None,:], np.sign(Xte)*np.log1p(np.abs(Xte)), Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Ztr = (Xtr-mu)/sd; Zte = (Xte-mu)/sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr),1))]); Zte = np.hstack([Zte, np.ones((len(Zte),1))])
    ytr = tr.future_spend_4w.to_numpy(float)
    p = Ztr.shape[1]
    A = Ztr.T@Ztr + alpha*np.eye(p); A[-1,-1] -= alpha
    w = np.linalg.solve(A, Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), **kw):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=fit_predict(tr,te,cols,a,**kw)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[0,0.3,1,3,10,30,100,300,1000]
print('E013 baseline alpha curve:', cv(m, feat, alphas))

# bias structure at alpha=3
tr=m[m.snapshot_day<431]; te=m[m.snapshot_day==431]
pr=fit_predict(tr,te,feat,3)
y=te.future_spend_4w.to_numpy(float)
qc=pd.qcut(pr,10,duplicates='drop')
g=pd.DataFrame({'p':pr,'y':y,'q':qc}).groupby('q',observed=True).agg(pmean=('p','mean'),ymean=('y','mean'),mae=('y',lambda s:s))
g['mae']=np.abs(pd.DataFrame({'p':pr,'y':y}).groupby(pd.qcut(pr,10,duplicates='drop').codes).apply(lambda d: np.mean(np.abs(d.p-d.y))))
print(g.round(1))
print('mean pred', pr.mean().round(1), 'mean y', y.mean().round(1), 'median pred', np.median(pr).round(1), 'median y', np.median(y).round(1))

# ---- cell ----
import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

# find duplicate / constant columns on train rows
sub = m[m.snapshot_day<347][feat]
nun = sub.nunique()
const = nun[nun<=1].index.tolist()
print('constant cols:', const)
corr = sub.corr().abs()
dup = [(a,b) for i,a in enumerate(feat) for b in feat[i+1:] if corr.loc[a,b]>0.999]
print('near-dup pairs:', dup[:40], '... total', len(dup))

def prep(tr, te, cols, alpha, logskew=False):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if logskew:
        sk = np.abs(pd.DataFrame(Xtr).skew().to_numpy())
        ms = sk > 2
        Xtr = np.where(ms[None,:], np.sign(Xtr)*np.log1p(np.abs(Xtr)), Xtr)
        Xte = np.where(ms[None,:], np.sign(Xte)*np.log1p(np.abs(Xte)), Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Ztr = (Xtr-mu)/sd; Zte = (Xte-mu)/sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr),1))]); Zte = np.hstack([Zte, np.ones((len(Zte),1))])
    ytr = tr.future_spend_4w.to_numpy(float)
    p = Ztr.shape[1]
    A = Ztr.T@Ztr + alpha*np.eye(p); A[-1,-1] -= alpha
    w = np.linalg.pinv(A) @ (Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), logskew=False):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=prep(tr,te,cols,a,logskew)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[0,1,3,10,30,100,300,1000]
print('E013 alpha curve:', cv(m, feat, alphas))
print('E013 alpha curve logskew:', cv(m, feat, alphas, logskew=True))

# ---- cell ----
import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

def prep(tr, te, cols, alpha, logskew=False, clip_q=None):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if clip_q is not None:
        lo = np.quantile(Xtr, clip_q, axis=0); hi = np.quantile(Xtr, 1-clip_q, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
    if logskew:
        sk = np.abs(pd.DataFrame(Xtr).skew().to_numpy())
        ms = sk > 2
        Xtr = np.where(ms[None,:], np.sign(Xtr)*np.log1p(np.abs(Xtr)), Xtr)
        Xte = np.where(ms[None,:], np.sign(Xte)*np.log1p(np.abs(Xte)), Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Ztr = (Xtr-mu)/sd; Zte = (Xte-mu)/sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr),1))]); Zte = np.hstack([Zte, np.ones((len(Zte),1))])
    ytr = tr.future_spend_4w.to_numpy(float)
    p = Ztr.shape[1]
    A = Ztr.T@Ztr + alpha*np.eye(p); A[-1,-1] -= alpha
    w = np.linalg.pinv(A) @ (Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), **kw):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=prep(tr,te,cols,a,**kw)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[1000,3000,10000,30000,100000]
print('E013 high alpha:', cv(m, feat, alphas))

# decile bias at alpha=1000
tr=m[m.snapshot_day<431]; te=m[m.snapshot_day==431]
pr=prep(tr,te,feat,1000); y=te.future_spend_4w.to_numpy(float)
qs=np.quantile(pr,np.linspace(0,1,11))
idx=np.clip(np.digitize(pr,qs[1:-1]),0,9)
for i in range(10):
    s=idx==i
    print(f'dec{i}: n={s.sum():5d} pred={pr[s].mean():7.1f} y={y[s].mean():7.1f} mae={np.abs(pr[s]-y[s]).mean():6.1f}')
print('overall pred mean', pr.mean().round(1), 'y mean', y.mean().round(1))

# prune dups/consts
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
print('pruned to', len(feat2))
print('pruned alpha curve:', cv(m, feat2, [100,1000,3000,10000]))
print('pruned+clip:', cv(m, feat2, [1000,3000], clip_q=0.01))
print('pruned+logskew:', cv(m, feat2, [1000,3000], logskew=True))

# ---- cell ----
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

view = agent_api.snapshot()
tx = view.transactions[['household_key','day','sales_value']]
days = np.arange(1,460)
piv = tx.pivot_table(index='household_key', columns='day', values='sales_value', aggfunc='sum').reindex(columns=days).fillna(0.0)
S = piv.to_numpy(); hh = piv.index.to_numpy(); hidx = {h:i for i,h in enumerate(hh)}
def wsum(a, lo, hi): return a[:, lo-1:hi].sum(axis=1)

grid = [67+28*k for k in range(0,13)]
pair_rows=[]
for t in grid:
    tr28 = wsum(S, t-27, t); nx28 = wsum(S, t+1, t+28)
    prev_act = wsum(S, max(1,t-83), t-28)
    for j in np.where(prev_act>0)[0]:
        pair_rows.append((hh[j], t, tr28[j], nx28[j]))
P = pd.DataFrame(pair_rows, columns=['h','t','trail','nxt'])
P['lx'] = np.log1p(P.trail)
print('pairs', P.shape)

def empirical_feats(D, h_list, nbins=40, k=300):
    Q = P[P.t <= D-28]
    lx = Q.lx.to_numpy(); nx = Q.nxt.to_numpy(); tq = Q.t.to_numpy()
    qs = np.quantile(lx, np.linspace(0,1,nbins+1)); qs[0]-=1; qs[-1]+=1
    b = np.clip(np.digitize(lx, qs[1:-1]),0,nbins-1)
    bin_med = np.array([np.median(nx[b==i]) if (b==i).sum() else np.nan for i in range(nbins)])
    out=[]
    for h in h_list:
        i = hidx.get(h)
        if i is None: out.append((np.nan,np.nan,np.nan)); continue
        t0 = wsum(S[i:i+1], D-27, D)[0]; x = np.log1p(t0)
        bi = np.clip(np.digitize([x], qs[1:-1])[0],0,nbins-1)
        d = np.abs(lx - x)
        nn = np.argpartition(d, min(k,len(d)-1))[:k]
        w = 0.5**((D-tq[nn])/730.0)
        out.append((bin_med[bi], np.median(nx[nn]), float(np.sum(w*nx[nn])/np.sum(w))))
    return pd.DataFrame(out, columns=['emp_bin_med','emp_knn_med','emp_knn_wmean'], index=h_list)

parts=[]
for D in sorted(m.snapshot_day.unique()):
    mm = m[m.snapshot_day==D]
    ef = empirical_feats(D, mm.household_key.tolist())
    ef['snapshot_day']=D
    parts.append(ef.reset_index())
ef_all = pd.concat(parts)
m2 = m.merge(ef_all, on=['household_key','snapshot_day'], how='left')
y=m2.future_spend_4w.to_numpy(float)
for c in ['emp_bin_med','emp_knn_med','emp_knn_wmean']:
    v=m2[c].to_numpy(float)
    print(c,'corr', round(np.corrcoef(np.nan_to_num(v),y)[0,1],3), 'standalone MAE', round(np.abs(np.nan_to_num(v)-y).mean(),1))
print('spend_84 standalone MAE', round(np.abs(m2.spend_84.to_numpy(float)-y).mean(),1))
print(m2[['emp_bin_med','emp_knn_med','emp_knn_wmean']].describe().round(1))

# local ridge CV with emp features added
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
base=[c for c in feat2]
print('pruned base:', cv(m2, base, [3000,10000]))
print('+emp:', cv(m2, base+['emp_bin_med','emp_knn_med','emp_knn_wmean'], [3000,10000]))
print('emp only:', cv(m2, ['emp_bin_med','emp_knn_med','emp_knn_wmean'], [0,1,10]))

# ---- cell ----
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

view = agent_api.snapshot()
tx = view.transactions[['household_key','day','sales_value']]
days = np.arange(1,460)
piv = tx.pivot_table(index='household_key', columns='day', values='sales_value', aggfunc='sum').reindex(columns=days).fillna(0.0)
S = piv.to_numpy(); hh = piv.index.to_numpy(); hidx = {h:i for i,h in enumerate(hh)}
def wsum(a, lo, hi): return a[:, lo-1:hi].sum(axis=1)

grid = [67+28*k for k in range(0,13)]
pair_rows=[]
for t in grid:
    tr28 = wsum(S, t-27, t); nx28 = wsum(S, t+1, t+28)
    prev_act = wsum(S, max(1,t-83), t-28)
    for j in np.where(prev_act>0)[0]:
        pair_rows.append((hh[j], t, tr28[j], nx28[j]))
P = pd.DataFrame(pair_rows, columns=['h','t','trail','nxt'])
P['lx'] = np.log1p(P.trail)

def empirical_feats(D, h_list, nbins=40, k=300):
    Q = P[P.t <= D-28]
    lx = Q.lx.to_numpy(); nx = Q.nxt.to_numpy(); tq = Q.t.to_numpy()
    qs = np.quantile(lx, np.linspace(0,1,nbins+1)); qs[0]-=1; qs[-1]+=1
    b = np.clip(np.digitize(lx, qs[1:-1]),0,nbins-1)
    bin_med = np.array([np.median(nx[b==i]) if (b==i).sum() else np.nan for i in range(nbins)])
    out=[]
    for h in h_list:
        i = hidx.get(h)
        if i is None: out.append((np.nan,np.nan,np.nan)); continue
        t0 = wsum(S[i:i+1], D-27, D)[0]; x = np.log1p(t0)
        bi = np.clip(np.digitize([x], qs[1:-1])[0],0,nbins-1)
        d = np.abs(lx - x)
        nn = np.argpartition(d, min(k,len(d)-1))[:k]
        w = 0.5**((D-tq[nn])/730.0)
        out.append((bin_med[bi], np.median(nx[nn]), float(np.sum(w*nx[nn])/np.sum(w))))
    return pd.DataFrame(out, columns=['emp_bin_med','emp_knn_med','emp_knn_wmean'],
                        index=pd.Index(h_list, name='household_key'))

parts=[]
for D in sorted(m.snapshot_day.unique()):
    mm = m[m.snapshot_day==D]
    ef = empirical_feats(D, mm.household_key.tolist())
    parts.append(ef.reset_index())
ef_all = pd.concat(parts, ignore_index=True)
m2 = m.merge(ef_all, on=['household_key','snapshot_day'], how='left')
y=m2.future_spend_4w.to_numpy(float)
for c in ['emp_bin_med','emp_knn_med','emp_knn_wmean']:
    v=m2[c].to_numpy(float)
    print(c,'corr', round(np.corrcoef(np.nan_to_num(v),y)[0,1],3), 'standalone MAE', round(np.abs(np.nan_to_num(v)-y).mean(),1))
print('spend_84 standalone MAE', round(np.abs(m2.spend_84.to_numpy(float)-y).mean(),1))

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
print('pruned base:', cv(m2, feat2, [3000,10000]))
print('+emp:', cv(m2, feat2+['emp_bin_med','emp_knn_med','emp_knn_wmean'], [3000,10000]))
print('emp only:', cv(m2, ['emp_bin_med','emp_knn_med','emp_knn_wmean'], [0,1,10]))

# ---- cell ----
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

view = agent_api.snapshot()
tx = view.transactions[['household_key','day','sales_value']]
days = np.arange(1,460)
piv = tx.pivot_table(index='household_key', columns='day', values='sales_value', aggfunc='sum').reindex(columns=days).fillna(0.0)
S = piv.to_numpy(); hh = piv.index.to_numpy(); hidx = {h:i for i,h in enumerate(hh)}
def wsum(a, lo, hi): return a[:, lo-1:hi].sum(axis=1)

grid = [67+28*k for k in range(0,13)]
pair_rows=[]
for t in grid:
    tr28 = wsum(S, t-27, t); nx28 = wsum(S, t+1, t+28)
    prev_act = wsum(S, max(1,t-83), t-28)
    for j in np.where(prev_act>0)[0]:
        pair_rows.append((hh[j], t, tr28[j], nx28[j]))
P = pd.DataFrame(pair_rows, columns=['h','t','trail','nxt'])
P['lx'] = np.log1p(P.trail)

def empirical_feats(D, h_list, nbins=40, k=300):
    Q = P[P.t <= D-28]
    lx = Q.lx.to_numpy(); nx = Q.nxt.to_numpy(); tq = Q.t.to_numpy()
    qs = np.quantile(lx, np.linspace(0,1,nbins+1)); qs[0]-=1; qs[-1]+=1
    b = np.clip(np.digitize(lx, qs[1:-1]),0,nbins-1)
    bin_med = np.array([np.median(nx[b==i]) if (b==i).sum() else np.nan for i in range(nbins)])
    out=[]
    for h in h_list:
        i = hidx.get(h)
        if i is None: out.append((np.nan,np.nan,np.nan)); continue
        t0 = wsum(S[i:i+1], D-27, D)[0]; x = np.log1p(t0)
        bi = np.clip(np.digitize([x], qs[1:-1])[0],0,nbins-1)
        d = np.abs(lx - x)
        nn = np.argpartition(d, min(k,len(d)-1))[:k]
        w = 0.5**((D-tq[nn])/730.0)
        out.append((bin_med[bi], np.median(nx[nn]), float(np.sum(w*nx[nn])/np.sum(w))))
    return pd.DataFrame(out, columns=['emp_bin_med','emp_knn_med','emp_knn_wmean'],
                        index=pd.Index(h_list, name='household_key'))

parts=[]
for D in sorted(m.snapshot_day.unique()):
    mm = m[m.snapshot_day==D]
    ef = empirical_feats(D, mm.household_key.tolist())
    ef['snapshot_day']=D
    parts.append(ef.reset_index())
ef_all = pd.concat(parts, ignore_index=True)
m2 = m.merge(ef_all, on=['household_key','snapshot_day'], how='left')
y=m2.future_spend_4w.to_numpy(float)
for c in ['emp_bin_med','emp_knn_med','emp_knn_wmean']:
    v=m2[c].to_numpy(float)
    print(c,'corr', round(np.corrcoef(np.nan_to_num(v),y)[0,1],3), 'standalone MAE', round(np.abs(np.nan_to_num(v)-y).mean(),1))
print('spend_84 standalone MAE', round(np.abs(m2.spend_84.to_numpy(float)-y).mean(),1))

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
print('pruned base:', cv(m2, feat2, [3000,10000]))
print('+emp:', cv(m2, feat2+['emp_bin_med','emp_knn_med','emp_knn_wmean'], [3000,10000]))
print('emp only:', cv(m2, ['emp_bin_med','emp_knn_med','emp_knn_wmean'], [0,1,10]))

# ---- cell ----
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

view = agent_api.snapshot()
tx = view.transactions[['household_key','day','sales_value']]
days = np.arange(1,460)
piv = tx.pivot_table(index='household_key', columns='day', values='sales_value', aggfunc='sum').reindex(columns=days).fillna(0.0)
S = piv.to_numpy(); hh = piv.index.to_numpy(); hidx = {h:i for i,h in enumerate(hh)}
def wsum(a, lo, hi): return a[:, lo-1:hi].sum(axis=1)

# cadence/phase features per snapshot
def cadence_feats(D, h_list):
    out=[]
    for h in h_list:
        i = hidx.get(h)
        if i is None: out.append((np.nan,)*6); continue
        row = S[i]
        act = np.where(row[:D]>0)[0] + 1  # purchase days
        if len(act)==0: out.append((np.nan,)*6); continue
        rec = D - act[-1]
        gaps = np.diff(act) if len(act)>1 else np.array([np.nan])
        gmed = np.median(gaps) if len(gaps)>0 else np.nan
        gmean = np.mean(gaps[-8:]) if len(gaps)>0 else np.nan
        # expected trips in next 28d given cadence g and recency r: trips at r+g, r+2g, ... <= 28
        def exp_trips(g, r):
            if not np.isfinite(g) or g<=0: return np.nan
            k = 0
            t = r + g
            while t <= 28 and k < 10: k += 1; t += g
            return k
        et_med = exp_trips(gmed, rec); et_mean = exp_trips(gmean, rec)
        bval = row[act[-1]-1]
        out.append((rec, gmed, gmean, et_med, et_mean, et_med*bval if np.isfinite(et_med) else np.nan))
    return pd.DataFrame(out, columns=['cd_rec','cd_gmed','cd_gmean','cd_et_med','cd_et_mean','cd_exp_spend'],
                        index=pd.Index(h_list, name='household_key'))
parts=[]
for D in sorted(m.snapshot_day.unique()):
    mm = m[m.snapshot_day==D]
    ef = cadence_feats(D, mm.household_key.tolist()); ef['snapshot_day']=D
    parts.append(ef.reset_index())
cad = pd.concat(parts, ignore_index=True)
m2 = m.merge(cad, on=['household_key','snapshot_day'], how='left')

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
print('base:', cv(m2, feat2, [10000]))
print('+cadence:', cv(m2, feat2+['cd_rec','cd_gmed','cd_gmean','cd_et_med','cd_et_mean','cd_exp_spend'], [10000]))

# error decomposition at alpha=10000, day 431
tr=m2[m2.snapshot_day<431]; te=m2[m2.snapshot_day==431]
pr=prep(tr,te,feat2,10000); y=te.future_spend_4w.to_numpy(float)
e=np.abs(pr-y)
z=y==0
print('y=0 rows: n=%.3f share, MAE=%.1f, mean pred=%.1f' % (z.mean(), e[z].mean(), pr[z].mean()))
print('y>0 rows: MAE=%.1f' % e[~z].mean())
b=pd.cut(y,[-1,0.01,25,75,150,300,1e5])
print(pd.DataFrame({'y':y,'p':pr,'e':e,'b':b}).groupby('b',observed=True).agg(n=('e','size'),mae=('e','mean'),pred=('p','mean'),y=('y','mean')))
# contribution to total MAE by band
gb=pd.DataFrame({'e':e,'b':b}).groupby('b',observed=True)['e'].sum()
print('MAE share by y-band:'); print((gb/gb.sum()).round(3))
# has_demo effect
hd = te.has_demo.to_numpy(float)
print('MAE with demo %.1f vs without %.1f' % (e[hd>0].mean(), e[hd==0].mean()))

# ---- cell ----
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

# ---- cell ----
import pandas as pd, numpy as np, agent_api
pd.set_option('mode.chained_assignment', None)

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

mk = agent_api.load_saved('e015_market_ctx.parquet')
mkc = [c for c in mk.columns if c not in ('household_key','snapshot_day') and c not in m.columns]
print('market cols:', mkc)
m3 = m.merge(mk[['household_key','snapshot_day']+mkc], on=['household_key','snapshot_day'], how='left')

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
print('E013 local:', cv(m3, feat2, [3000,10000]))
print('E015 local (+market):', cv(m3, feat2+mkc, [3000,10000]))

# group ablation at alpha=3000
groups = {
 'demo': [c for c in feat2 if c.startswith(('age_ord','income_ord','size_ord','grp_ord','kid_ord','c2_','ho_','kid_','has_demo','ix_'))],
 'denoise_c': [c for c in feat2 if c.startswith(('c_','z_'))],
 'nrank': [c for c in feat2 if c.startswith('n_')],
 'weekly': [c for c in feat2 if c.startswith(('wk','w3','w4','w5','w6','w7','w8'))],
 'fwd28': [c for c in feat2 if c.startswith(('fwd28','log_fwd28','ewma','log_ewma'))],
 'calendar': [c for c in feat2 if c in ('day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx')],
 'basket': [c for c in feat2 if c.startswith(('last_b','bmean','bstd','bmax','trips28b','trips84b','qty28','spend28b','weekend','evening','morning','last_ratio','stockup','bcv','unit_price','log_last'))],
 'longrun_x': [c for c in feat2 if c.startswith('x_')],
 'gaps': [c for c in feat2 if c.startswith(('gap_','recency','tenure'))],
}
base_cv = cv(m3, feat2, [3000])[3000]
print('base', base_cv)
for g, cols in groups.items():
    cols=[c for c in cols if c in feat2]
    rest=[c for c in feat2 if c not in cols]
    v = cv(m3, rest, [3000])[3000]
    print(f'drop {g:10s} (n={len(cols):3d}): {v}  delta {v-base_cv:+.3f}')

# hinge basis on log spend
H = m3.copy()
for src in ['spend_84','spend_28','fwd28_mean']:
    lv = np.log1p(np.maximum(H[src].fillna(0).to_numpy(float),0))
    qs = np.quantile(lv, [0.2,0.4,0.6,0.8])
    for j,q in enumerate(qs):
        H[f'hg_{src}_{j}'] = np.maximum(lv-q, 0)
hgs=[c for c in H.columns if c.startswith('hg_')]
print('E013+hinges:', cv(H, feat2+mkc+hgs, [3000,10000]))

# ---- cell ----
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

mk = agent_api.load_saved('e015_market_ctx.parquet')
mkc = [c for c in mk.columns if c not in ('household_key','snapshot_day') and c not in m.columns]
m3 = m.merge(mk[['household_key','snapshot_day']+mkc], on=['household_key','snapshot_day'], how='left')

# constant median reference on train rows
ytr_all = m[m.snapshot_day<459].future_spend_4w.to_numpy(float)
med_all = np.median(ytr_all)
print('global median', med_all, 'const-median MAE (train rows):', round(np.abs(ytr_all-med_all).mean(),2))
# per-snapshot-day median predictor (uses only train info at prediction time? no - uses target of same day; just context)
for D in [347,375,403,431]:
    te=m[m.snapshot_day==D]
    tr=m[m.snapshot_day<D]
    print(D, 'median-of-train MAE', round(np.abs(te.future_spend_4w-np.median(tr.future_spend_4w)).mean(),2))

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

# hinge features (transforms of existing columns - computable on saved table)
H = m3.copy()
hgs=[]
for src in ['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_56']:
    lv = np.log1p(np.maximum(H[src].fillna(0).to_numpy(float),0))
    qs = np.quantile(lv, [0.15,0.3,0.45,0.6,0.75,0.9])
    for j,q in enumerate(qs):
        H[f'hg_{src}_{j}'] = np.maximum(lv-q, 0); hgs.append(f'hg_{src}_{j}')
cal = ['day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx']
cal = [c for c in cal if c in feat2]
longrun_x = [c for c in feat2 if c.startswith('x_')]
weekly = [c for c in feat2 if c.startswith(('wk','w3','w4','w5','w6','w7','w8'))]
nrank = [c for c in feat2 if c.startswith('n_')]

F = feat2+mkc
print('E015      :', cv(m3, F, [3000,10000]))
print('E015+hg   :', cv(m3, F+hgs, [3000,10000]))
print('E015+hg-cal:', cv(m3, [c for c in F+hgs if c not in cal], [3000,10000]))
print('E015+hg-cal-lx:', cv(m3, [c for c in F+hgs if c not in cal+longrun_x], [3000,10000]))
print('E015+hg-cal-wk:', cv(m3, [c for c in F+hgs if c not in cal+weekly], [3000,10000]))
print('E015+hg-cal-lx-wk:', cv(m3, [c for c in F+hgs if c not in cal+longrun_x+weekly], [3000,10000]))
print('E015-cal  :', cv(m3, [c for c in F if c not in cal], [3000,10000]))
print('E013+hg-cal:', cv(m3, [c for c in feat2+hgs if c not in cal], [3000,10000]))

# ---- cell ----
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
mk = agent_api.load_saved('e015_market_ctx.parquet')
mkc = [c for c in mk.columns if c not in ('household_key','snapshot_day') and c not in m.columns]
m3 = m.merge(mk[['household_key','snapshot_day']+mkc], on=['household_key','snapshot_day'], how='left')

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

H = m3.copy()
hgs=[]
for src in ['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_56']:
    lv = np.log1p(np.maximum(H[src].fillna(0).to_numpy(float),0))
    qs = np.quantile(lv, [0.15,0.3,0.45,0.6,0.75,0.9])
    for j,q in enumerate(qs):
        H[f'hg_{src}_{j}'] = np.maximum(lv-q, 0); hgs.append(f'hg_{src}_{j}')
rawcal = [c for c in ['day_idx','week_of_year','month_idx'] if c in feat2]
allcal = rawcal + [c for c in ['sin1','cos1','sin2','cos2'] if c in feat2]
longrun_x = [c for c in feat2 if c.startswith('x_')]
F = feat2 + mkc
print('rawcal cols:', rawcal)
print('A E015+hg            :', cv(H, F+hgs, [3000,10000]))
print('B E015+hg -rawcal    :', cv(H, [c for c in F+hgs if c not in rawcal], [3000,10000]))
print('C E015+hg -allcal    :', cv(H, [c for c in F+hgs if c not in allcal], [3000,10000]))
print('D E013+hg -rawcal    :', cv(H, [c for c in feat2+hgs if c not in rawcal], [3000,10000]))
print('E E015+hg -rc-lx     :', cv(H, [c for c in F+hgs if c not in rawcal+longrun_x], [3000,10000]))
print('F E015 -rawcal       :', cv(H, [c for c in F if c not in rawcal], [3000,10000]))
# per-day for best candidate vs E015
for name, cols in [('E015', F), ('E015+hg-rc', [c for c in F+hgs if c not in rawcal])]:
    per=[]
    for D in (347,375,403,431):
        tr=H[H.snapshot_day<D]; te=H[H.snapshot_day==D]
        per.append(round(float(np.mean(np.abs(prep(tr,te,cols,10000)-te.future_spend_4w.to_numpy(float)))),2))
    print(name, 'per-day', per)

# ---- cell ----
import pandas as pd, numpy as np, agent_api

base = agent_api.load_saved('e015_market_ctx.parquet')
print('base shape', base.shape)
cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
# drop day_idx and x_* long-run features
drop_cols = [c for c in cols if c == 'day_idx' or c.startswith('x_')]
print('dropping', len(drop_cols), drop_cols[:25])
T = base.drop(columns=drop_cols).copy()
# add hinge features on log1p of key spend columns
hgs=[]
for src in ['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_56']:
    lv = np.log1p(np.maximum(T[src].fillna(0).to_numpy(float),0))
    qs = np.quantile(lv, [0.15,0.3,0.45,0.6,0.75,0.9])
    for j,q in enumerate(qs):
        T[f'hg_{src}_{j}'] = np.maximum(lv-q, 0); hgs.append(f'hg_{src}_{j}')
print('final shape', T.shape, 'n_feat', T.shape[1]-2)
print('rows per day:'); print(T.snapshot_day.value_counts().sort_index().to_string())
assert T.shape[1]-2 <= 500
p = agent_api.save_table(T, 'e018_hinge_prune')
print('saved', p)