
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
Xtr = prep(tr).fillna(prep(tr).median()); Xva = prep(va)
med = prep(tr).median(); Xtr = prep(tr).fillna(med); Xva = prep(va).fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
X1 = np.c_[np.ones(len(Xtr_s)), Xtr_s]; Xv1 = np.c_[np.ones(len(Xva_s)), Xva_s]
def ridge_fit(X,y,lam):
    A = X.T@X + lam*np.eye(X.shape[1]); A[0,0]-=lam
    return np.linalg.solve(A, X.T@y)
for lam in [0.3,1,3,10,30,100,300,1000]:
    b = ridge_fit(X1,ytr,lam)
    print('lam %6.1f  val MAE %.3f' % (lam, np.abs(Xv1@b-yva).mean()))
b = ridge_fit(X1,ytr,30)
pred = Xv1@b
print('\nlam30 val MAE: %.3f' % np.abs(pred-yva).mean())
for d in sorted(va.snapshot_day.unique()):
    k = (va.snapshot_day==d).values
    print(' day %d: mean y %7.1f  mean pred %7.1f  MAE %.1f' % (d, yva[k].mean(), pred[k].mean(), np.abs(pred[k]-yva[k]).mean()))
res = ytr - X1@b
rc = pd.Series({f: np.corrcoef(Xtr[f], res)[0,1] for f in feats}).sort_values()
print('\nresid corr bottom10:'); print(rc.head(10).round(3).to_string())
print('resid corr top10:'); print(rc.tail(10).round(3).to_string())
# MAE by predicted decile on val
q = pd.qcut(pred, 10, duplicates='drop')
print('\nval MAE by pred decile:')
print(pd.DataFrame({'pred':pred,'y':yva,'q':q}).groupby('q').apply(lambda g: pd.Series({'n':len(g),'mean_pred':g.pred.mean(),'mean_y':g.y.mean(),'mae':np.abs(g.pred-g.y).mean()})).round(1).to_string())
