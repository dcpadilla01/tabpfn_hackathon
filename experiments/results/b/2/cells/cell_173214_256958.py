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