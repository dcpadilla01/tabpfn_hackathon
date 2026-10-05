import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
tmp = pd.DataFrame({'y':y,'absres':np.abs(resid),'pred':pred})
q = pd.qcut(y, 10, duplicates='drop')
print(tmp.groupby(q, observed=True).agg(n=('y','size'), mae=('absres','mean'), ymean=('y','mean'), pmean=('pred','mean')).round(1))

num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
rows=[]
for c in num:
    a = df[c].astype(float).values
    m = np.isfinite(a)
    if m.sum()<1000: continue
    med = np.nanmedian(a)
    aa = np.where(m, a, med)
    if aa.std()==0: continue
    r = np.corrcoef(aa, resid)[0,1]
    rows.append((abs(r), r, c))
rows.sort(reverse=True)
print("\nTop |corr| with residual (blend):")
for _,r,c in rows[:25]: print(f"  {r:+.3f} {c}")
