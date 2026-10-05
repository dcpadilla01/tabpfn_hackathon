
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

# Diagnostic at snapshot 431 (allowed view) + train targets
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')
print("tx max day:", tx['day'].max(), "n:", len(tx))
t = agent_api.train_targets()
y431 = t[t.snapshot_day==431].set_index('household_key')['future_spend_4w']
hh = y431.index

def win_spend(tx, hh, lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)&(tx.household_key.isin(hh))]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)

s28  = win_spend(tx, hh, 431-28, 431)
s56  = win_spend(tx, hh, 431-56, 431)
s84  = win_spend(tx, hh, 431-84, 431)
s364 = win_spend(tx, hh, 431-364, 431-336)   # same 4-week window one year ago
s336 = win_spend(tx, hh, 431-392, 431-364)   # window 1yr+4wk ago
y = y431.values
def corr(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float); m=~np.isnan(a)&~np.isnan(b)
    return np.corrcoef(a[m],b[m])[0,1]
print("corr y~s28:", round(corr(s28,y),3), " y~s56:", round(corr(s56,y),3),
      " y~s84:", round(corr(s84,y),3), " y~s364:", round(corr(s364,y),3), " y~s336:", round(corr(s336,y),3))

# partial: does s364 add beyond s28+s84? quick 2-feature OOF-ish check via binning
df = pd.DataFrame({'y':y,'s28':s28.values,'s84':s84.values,'s364':s364.values,'s336':s336.values})
df['s28b'] = pd.qcut(df.s28.rank(method='first'), 5, labels=False, duplicates='drop')
df['s364b'] = pd.qcut(df.s364.rank(method='first'), 5, labels=False, duplicates='drop')
print(df.groupby(['s28b','s364b'], observed=True)['y'].mean().unstack().round(1))

# zero structure
print("\nP(y==0 | s28==0):", round((df.loc[df.s28==0,'y']==0).mean(),3), " n:", (df.s28==0).sum())
print("P(y==0 | s28>0):", round((df.loc[df.s28>0,'y']==0).mean(),3))
print("mean y | s28==0:", round(df.loc[df.s28==0,'y'].mean(),2), " mean y | s28>0:", round(df.loc[df.s28>0,'y'].mean(),2))
print("mean s364 | s28==0:", round(df.loc[df.s28==0,'s364'].mean(),2))
print("mean y | s28==0 & s364>50:", round(df.loc[(df.s28==0)&(df.s364>50),'y'].mean(),2), " n:", ((df.s28==0)&(df.s364>50)).sum())
