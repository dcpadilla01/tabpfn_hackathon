
import agent_api, pandas as pd, numpy as np
# rebuild d quickly (cache-free but cheap ops only) - reuse previous run's logic minimally
tt = agent_api.train_targets()
base = agent_api.load_saved('e011_rank.parquet')
print('base index name:', base.index.name, base.index[:3])
print('tt head:'); print(tt.head(3))
# check alignment: does base row i correspond to tt row i?
b = base.reset_index(drop=True)
m = tt.merge(b, left_index=True, right_index=True)
y = m.future_spend_4w.values
print('corr(spend_28, y) =', np.corrcoef(m.spend_28.fillna(0), y)[0,1])
print('MAE pred=spend_28 on day431:', np.abs(m.spend_28[m.snapshot_day==431].fillna(0).values - y[m.snapshot_day==431]).mean())
print('MAE pred=0.9*spend_28:', np.abs(0.9*m.spend_28[m.snapshot_day==431].fillna(0).values - y[m.snapshot_day==431]).mean())
# tiny ridge on just spend_28
tr = (m.snapshot_day<=403).values; va=(m.snapshot_day==431).values
X = m.spend_28.fillna(0).values
mu,sd = X[tr].mean(), X[tr].std()
Xz = ((X-mu)/sd).reshape(-1,1)
A = np.hstack([np.ones((len(Xz),1)), Xz])
w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
print('ridge spend28 only coefs', w, 'MAE', np.abs(A[va]@w - y[va]).mean())
# now base numeric only, check per-column NaN rates and std
num = m.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
print('n num cols', len(num))
nn = num.isna().mean().sort_values(ascending=False)
print('top NaN cols:'); print(nn.head(8))
print('cols with zero std:', (num.fillna(0).std()==0).sum())
