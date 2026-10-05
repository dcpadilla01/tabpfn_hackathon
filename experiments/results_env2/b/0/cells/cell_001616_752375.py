
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
Xtr, Xva = X[trm], X[vam]
med = Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
A = np.c_[np.ones(trm.sum()), ((Xtr-mu)/sd).values]; B = np.c_[np.ones(vam.sum()), ((Xva-mu)/sd).values]
lam = 30
M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
b = np.linalg.solve(M, A.T@y[trm])
pred = B@b; yva = y[vam]
print('pseudo-val MAE %.2f' % np.abs(pred-yva).mean())
dv = m.snapshot_day[vam]
for d in sorted(dv.unique()):
    k = (dv==d).values
    print(' day %d: n %d  mean y %7.1f mean pred %7.1f  MAE %.1f' % (d,k.sum(),yva[k].mean(),pred[k].mean(),np.abs(pred[k]-yva[k]).mean()))
z = yva==0
print('\nzero rows: n=%d (%.1f%%)  MAE %.1f  mean pred %.1f' % (z.sum(),100*z.mean(),np.abs(pred[z]-yva[z]).mean(),pred[z].mean()))
print('nonzero rows: MAE %.1f' % np.abs(pred[~z]-yva[~z]).mean())
print('share of total MAE from zero rows: %.2f' % (np.abs(pred[z]-yva[z]).sum()/np.abs(pred-yva).sum()))
res = y[trm] - A@b
Xtr_df = pd.DataFrame(Xtr, columns=feats)
rc = Xtr_df.corrwith(pd.Series(res)).sort_values()
print('\nresid corr bottom12:'); print(rc.head(12).round(3).to_string())
print('resid corr top12:'); print(rc.tail(12).round(3).to_string())
q = pd.qcut(yva, 10, duplicates='drop')
g = pd.DataFrame({'pred':pred,'y':yva,'q':q}).groupby('q',observed=True).apply(lambda t: pd.Series({'n':len(t),'mean_y':t.y.mean(),'mean_pred':t.pred.mean(),'mae':np.abs(t.pred-t.y).mean()}))
print('\nMAE by true-spend decile:'); print(g.round(1).to_string())
