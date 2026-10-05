
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = df[feats].copy()
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype(float)
trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
med = X[trm].median(); Xf = X.fillna(med)

def proxy(Xnew, lam=30):
    mu, sd = Xnew[trm].mean(), Xnew[trm].std().replace(0,1)
    A = np.c_[np.ones(trm.sum()), ((Xnew[trm]-mu)/sd).values]
    B = np.c_[np.ones(vam.sum()), ((Xnew[vam]-mu)/sd).values]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm])
    return np.abs(B@b - y[vam]).mean()

base = proxy(Xf); print('base E013 proxy: %.3f' % base)

def hinge(s, t): return np.maximum(s - t, 0)
def add_hinges(Xnew, spec):
    for col, thrs in spec.items():
        for t in thrs:
            Xnew[col+'_h%d'%t] = np.maximum(Xnew[col] - t, 0)
    return Xnew

specA = {'dec_28':[25,75,150,300,600], 'dec_56':[25,75,150,300,600], 'spend_84':[50,150,400,800],
         'z_rate84':[25,75,200], 'weekly_mean_12':[25,75,200]}
print('A hinges tail: %.3f' % proxy(add_hinges(Xf.copy(), specA)))
specB = {'dec_28':[10,25,50,100,200,400], 'spend_84':[25,75,200,500,1000]}
print('B hinges dec28+sp84: %.3f' % proxy(add_hinges(Xf.copy(), specB)))
# winsorize heavy spend cols at train p99
Xw = Xf.copy()
sp_cols = [c for c in Xf.columns if c.startswith(('spend','dec_','weekly','avg_'))]
for c in sp_cols:
    Xw[c] = Xw[c].clip(upper=Xw[c][trm].quantile(0.99))
print('C winsorize: %.3f' % proxy(Xw))
print('D winsor+hingesB: %.3f' % proxy(add_hinges(Xw.copy(), specB)))
# log versions of spend cols
Xl = Xf.copy()
for c in sp_cols:
    Xl[c] = np.log1p(Xl[c])
print('E log spend cols: %.3f' % proxy(Xl))
print('F log+hingesB: %.3f' % proxy(add_hinges(Xl.copy(), specB)))
