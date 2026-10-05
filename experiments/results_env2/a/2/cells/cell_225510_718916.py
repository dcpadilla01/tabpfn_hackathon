import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.isotonic import IsotonicRegression
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
tr = df[df.snapshot_day <= 403]; va = df[df.snapshot_day == 431]
yv = va[TARGET].values

def fit_med(seed, X, y, n=1200, lr=0.03, md=7, mcw=10):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        random_state=seed, n_jobs=8, tree_method='hist')
    m.fit(X, y); return m

t0=time.time()
pa = np.mean([fit_med(s, tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2,3]], axis=0)
print('fit time', round(time.time()-t0,1), 'base MAE 431:', round(np.abs(pa-yv).mean(),3))

# honest cross-fit of isotonic median calibration on day 431
rng = np.random.RandomState(0); idx = rng.permutation(len(va))
h1, h2 = idx[:len(idx)//2], idx[len(idx)//2:]
def calibrate(ptrain, ytrain, ptest):
    iso = IsotonicRegression(y_min=0, out_of_bounds='clip')
    iso.fit(ptrain, ytrain)
    return iso.predict(ptest)
pc2 = calibrate(pa[h1], yv[h1], pa[h2])
print('cross-fit recal MAE (half2): base', round(np.abs(pa[h2]-yv[h2]).mean(),3),
      '-> recal', round(np.abs(pc2-yv[h2]).mean(),3))
# bias after recal
print('half2 bias base', round((pa[h2]-yv[h2]).mean(),1), 'recal', round((pc2-yv[h2]).mean(),1))

# full calibration curve from day 431 (for later use on validation)
iso_full = IsotonicRegression(y_min=0, out_of_bounds='clip'); iso_full.fit(pa, yv)
print('cal curve samples:', [(round(x,0), round(float(iso_full.predict([x])[0]),0)) for x in [0,10,25,50,100,200,400,800,1100]])

# zero classifier
from xgboost import XGBClassifier
yz = (tr[TARGET]==0).astype(int)
clf = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, random_state=1, n_jobs=8, tree_method='hist',
                    eval_metric='logloss')
clf.fit(tr[FEATS], yz)
p0 = clf.predict_proba(va[FEATS])[:,1]
from sklearn.metrics import roc_auc_score
print('zero-classifier AUC 431:', round(roc_auc_score(yz.values if hasattr(yz,'values') else yz, p0),3))
print('base rate zero:', round(yz.mean(),3), 'mean p0 on 431:', round(p0.mean(),3))
for thr in [0.4,0.5,0.6]:
    pf = np.where(p0>thr, 0.0, pa)
    print(f'zero-override thr={thr}: MAE', round(np.abs(pf-yv).mean(),3))
# both combined (recal trained on full 431 = optimistic for 431, just indicative)
pc_all = iso_full.predict(pa)
pf2 = np.where(p0>0.5, 0.0, pc_all)
print('recal+override (in-sample cal, indicative):', round(np.abs(pf2-yv).mean(),3))
