
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr); Xva = prep(va)
med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
Xtr_s = np.c_[np.ones(len(Xtr_s)), Xtr_s]; Xva_s = np.c_[np.ones(len(Xva_s)), Xva_s]
def ridge_fit(X,y,lam):
    A = X.T@X + lam*np.eye(X.shape[1]); A[0,0]-=lam
    return np.linalg.solve(A, X.T@y)
for lam in [0.3,1,3,10,30,100,300,1000]:
    b = ridge_fit(Xtr_s,ytr,lam)
    print('lam %6.1f  val MAE %.3f' % (lam, np.abs(Xva_s@b-yva).mean()))
b = ridge_fit(Xtr_s,ytr,30)
pred = Xva_s@b
print('\nval MAE lam30: %.3f' % np.abs(pred-yva).mean())
for d in sorted(va.snapshot_day.unique()):
    k = (va.snapshot_day==d).values
    print(' day %d: mean y %.1f mean pred %.1f' % (d, yva[k].mean(), pred[k].mean()))
res = ytr - Xtr_s@b
rc = pd.Series({f: np.corrcoef(Xtr[f], res)[0,1] for f in feats}).sort_values()
print('\nresid corr bottom10:'); print(rc.head(10).round(3).to_string())
print('resid corr top10:'); print(rc.tail(10).round(3).to_string())
