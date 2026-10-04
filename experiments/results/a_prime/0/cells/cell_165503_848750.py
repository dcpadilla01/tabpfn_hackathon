import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
m = A.load_saved('e013_stock.parquet')
snap_days=[431,459]; out={}
for sd in snap_days:
    v=A.snapshot(min(sd,459)); tr=v.transactions; tr=tr[tr.day<=sd]
    hh_codes,hh_uniq=pd.factorize(tr.household_key)
    day=tr.day.values; sv=tr.sales_value.values
    f=pd.DataFrame(index=hh_uniq)
    w=np.exp(np.log(0.5)*(sd-day)/28)
    s=np.zeros(len(hh_uniq)); np.add.at(s,hh_codes,sv*w); f['ew28']=s
    out[sd]=f
    print(sd,'index name:',f.index.name)
blk=pd.concat([out[sd].reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=sd) for sd in snap_days])
print(blk.columns.tolist(), blk.head(3))
df=m.merge(blk,on=['household_key','snapshot_day'],how='left')
print('ew28 notna:', df.ew28.notna().sum(), 'rows', len(df))
