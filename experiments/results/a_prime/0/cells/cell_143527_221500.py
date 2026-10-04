
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tx = agent_api.snapshot(459).transactions[['household_key','day','sales_value']]
tt = agent_api.train_targets()
s = agent_api.snapshot_days()['train']
def bs(lo,hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
rows=[]
for sd in s:
    cur = bs(sd-27, sd).rename('cur')
    yr  = bs(sd-391, sd-364).rename('yr')
    d = pd.concat([cur, yr], axis=1)
    d['snapshot_day']=sd
    rows.append(d.reset_index())
F = pd.concat(rows)
F = tt.merge(F, on=['household_key','snapshot_day'], how='inner').dropna(subset=['cur','yr'])
print('rows with lag364:', len(F))
X = np.column_stack([np.ones(len(F)), np.log1p(F.cur), np.log1p(F.yr)])
y = np.log1p(F.future_spend_4w)
b,_,_,_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X@b
b2,_,_,_ = np.linalg.lstsq(X[:,:2], y, rcond=None)
yhat2 = X[:,:2]@b2
print('coef [const, log cur, log yr]:', b.round(3))
print('R2 log with yr: %.3f  without: %.3f' % (1-((y-yhat)**2).mean()/((y-y.mean())**2).mean(), 1-((y-yhat2)**2).mean()/((y-y.mean())**2).mean()))
rc = np.corrcoef(np.log1p(F.cur), y)[0,1]; ry=np.corrcoef(np.log1p(F.yr), y)[0,1]; rcc=np.corrcoef(np.log1p(F.cur), np.log1p(F.yr))[0,1]
print('partial corr yr|cur: %.3f' % ((ry-rc*rcc)/np.sqrt((1-rc**2)*(1-rcc**2))))
