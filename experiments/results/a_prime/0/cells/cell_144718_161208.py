print("snapshot_days:", snapshot_days())
s = snapshot()
print("txn rows<=459:", s.transactions.shape)

base = load_saved('e005_full_plus_mix.parquet')
print("base shape:", base.shape)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print("n base feature cols:", len(bcols))
print("base cols:", bcols)

mix = load_saved('mix_v1.parquet')
print("mix cols:", [c for c in mix.columns if c not in ('household_key','snapshot_day')])

tt = train_targets()
print("targets shape:", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))

m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.astype(float)
rows=[]
for c in bcols:
    x = m[c]
    if x.dtype.kind not in 'ifb': continue
    xm = x.fillna(x.median()).astype(float)
    if xm.std()==0: continue
    rows.append((c, float(np.corrcoef(xm, y)[0,1])))
cs = pd.DataFrame(rows, columns=['f','r'])
cs['a']=cs.r.abs()
cs=cs.sort_values('a',ascending=False)
print("TOP CORR:")
print(cs.head(35).to_string())
print("BOTTOM CORR:")
print(cs.tail(8).to_string())