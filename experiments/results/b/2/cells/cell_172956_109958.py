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