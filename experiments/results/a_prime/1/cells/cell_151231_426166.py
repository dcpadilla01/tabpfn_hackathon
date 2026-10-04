import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_transforms.parquet')

m = t.merge(E3[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','spend_l123_mean']], on=KEY, how='left')
m = m.merge(NF[['household_key','snapshot_day','nf_lspend123','nf_lspend1']], on=KEY, how='left')
print("rows", len(m))
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean','nf_lspend1','nf_lspend123']:
    v=m[c]; print(c, "nan:",v.isna().sum(), "zero:",(v==0).sum(), "med:",np.nanmedian(v))

# is spend_l123_mean the mean or sum?
chk = (m.spend_l1+m.spend_l2+m.spend_l3)
print("corr mean vs sum/3:", np.corrcoef(m.spend_l123_mean, chk/3)[0,1])

lg_mean = np.log1p(m.spend_l123_mean)
lg_sum  = np.log1p(chk)
print("corr lg_mean vs lg_sum:", np.corrcoef(lg_mean, lg_sum)[0,1])
print("corr lg_mean vs nf_lspend123:", np.corrcoef(lg_mean, m.nf_lspend123)[0,1])

def uni(col_vals, y, days):
    fit=days<=347; val=days>=375
    x=col_vals.astype(float)
    mu=np.nanmean(x[fit]); sd=np.nanstd(x[fit]); sd=sd if sd>1e-9 else 1
    Z=np.where(np.isfinite(x),(x-mu)/sd,0.0)
    Zf=np.hstack([Z[fit,None],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val,None],np.ones((val.sum(),1))])
    A=Zf.T@Zf; b=Zf.T@y[fit]; p=2
    best=[]
    for lam in [0.01,0.1,1,10,100]:
        w=np.linalg.solve(A+lam*np.eye(p),b); best.append((np.abs(Zv@w-y[val]).mean(),lam))
    best.sort(); return best[0]

y=m.future_spend_4w.values.astype(float); d=m.snapshot_day.values
print("uni log1p(E3 mean):", uni(lg_mean.values,y,d))
print("uni log1p(sum)    :", uni(lg_sum.values,y,d))
print("uni nf_lspend123  :", uni(m.nf_lspend123.values,y,d))
print("uni nf_lspend1    :", uni(m.nf_lspend1.values,y,d))
print("uni log1p(E3 l1)  :", uni(np.log1p(m.spend_l1).values,y,d))
# where do they differ?
diff = (np.abs(lg_sum - m.nf_lspend123) > 0.01) & lg_sum.notna() & m.nf_lspend123.notna()
print("n differing rows:", diff.sum())
if diff.sum():
    print(m.loc[diff, ['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','nf_lspend1','nf_lspend123']].head(10))
# NaN handling: rows where E3 spend_l123_mean is NaN
nn = m.spend_l123_mean.isna()
print("E3 mean NaN rows:", nn.sum(), " their nf_lspend123 values:", m.loc[nn,'nf_lspend123'].unique()[:5], " mean y:", y[nn.values].mean())