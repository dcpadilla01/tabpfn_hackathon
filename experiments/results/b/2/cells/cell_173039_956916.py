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