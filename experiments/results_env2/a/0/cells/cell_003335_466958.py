import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import brier_score_loss

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=np.array([0.30,0.40,0.50,0.60,0.70])
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps,0),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| brier:', round(brier_score_loss((ye==0).astype(int), p0),4))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
# ablation: quantile-mapping without p0 (i.e., p0=0): qn=0.5 -> q50
twoC = qpreds[0.50]
print('positives-only q50 (no zero rule) MAE:', round(np.abs(twoC-ye).mean(),3))
b = pd.cut(ye, [-1,0,25,75,150,300,1e9])
df = pd.DataFrame({'ye':ye,'two':two,'p0':p0,'b':b})
print(df.groupby('b', observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_p=('two','median'), mae=('ye', lambda s: np.abs(s-df.loc[s.index,'two']).mean()), meanp0=('p0','mean')))
