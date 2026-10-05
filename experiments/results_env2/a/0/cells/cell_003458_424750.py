import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

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
# proper baseline: q50 on ALL rows, 3-model elementwise mean
ps_all=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); ps_all.append(m.predict(Xe))
base=np.clip(np.mean(ps_all,0),0,None)
print('baseline 3-model q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3))

Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE:', round(np.abs(two-ye).mean(),3))
# variant A: no hard zero, pure mixture scaling: (1-p0)*q_mapped
mixA=(1-p0)*Q[np.arange(len(ye)),lo]*(1-w)+ (1-p0)*Q[np.arange(len(ye)),hi]*w
print('variant A (1-p0)*q_mapped MAE:', round(np.abs(mixA-ye).mean(),3))
# variant B: two-part but threshold sweep
for thr in [0.4,0.45,0.5,0.55,0.6]:
    t=np.where(p0>=thr, 0.0, two)
    print(f'two-part thr={thr} MAE:', round(np.abs(t-ye).mean(),3))
# variant C: two-part with qn = 0.5 - 0.5*p0 (gentler mapping)
qn2=np.clip(0.5-0.5*p0, qgrid[0], qgrid[-1])
idx2=np.clip(np.searchsorted(qgrid, qn2),0,len(qgrid)-1)
lo2=np.clip(idx2-1,0,len(qgrid)-1); hi2=idx2
w2=np.where(qgrid[hi2]>qgrid[lo2],(qn2-qgrid[lo2])/np.maximum(qgrid[hi2]-qgrid[lo2],1e-9),0)
twoC=Q[np.arange(len(ye)),lo2]*(1-w2)+Q[np.arange(len(ye)),hi2]*w2
twoC=np.where(zero_mask,0.0,twoC)
print('variant C (qn=0.5-0.5p0) MAE:', round(np.abs(twoC-ye).mean(),3))
