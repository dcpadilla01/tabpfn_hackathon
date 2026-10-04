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