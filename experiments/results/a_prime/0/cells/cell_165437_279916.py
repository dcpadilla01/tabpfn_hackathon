import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
m = A.load_saved('e013_stock.parquet')
# quick rebuild of just EWMA at 431 to test merge
sd=431
v=A.snapshot(459); tr=v.transactions; tr=tr[tr.day<=sd]
hh_codes,hh_uniq=pd.factorize(tr.household_key)
day=tr.day.values; sv=tr.sales_value.values
w=np.exp(np.log(0.5)*(sd-day)/28)
s=np.zeros(len(hh_uniq)); np.add.at(s,hh_codes,sv*w)
f=pd.DataFrame({'household_key':hh_uniq,'snapshot_day':sd,'ew28x':s})
print(f.head(3))
mm=m[m.snapshot_day==sd].merge(f,on=['household_key','snapshot_day'],how='left')
print('non-null ew28x:', mm.ew28x.notna().sum(), 'of', len(mm))
print(mm[['household_key','spend_28','ew28x']].head())
print('corr', mm[['spend_28','ew28x']].corr().iloc[0,1])
