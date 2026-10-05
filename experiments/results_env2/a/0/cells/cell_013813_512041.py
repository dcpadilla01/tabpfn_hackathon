
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')

tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
val = f[f.snapshot_day.isin(A.snapshot_days()['validation'])]
def X(df): return df[feat_cols].values.astype(np.float32)

BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
UP = [0.1, 0.2, 0.3, 0.4, 0.5]   # low quantiles of positive part

def fit_q(Xtr, ytr, u):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr); return m

def run_fold(fit_max, eval_days):
    fi = tr.snapshot_day <= fit_max; ei = tr.snapshot_day.isin(eval_days)
    Xf, yf = X(tr[fi]), tr.future_spend_4w.values[fi]
    Xe, ye = X(tr[ei]), tr.future_spend_4w.values[ei]
    res = {}; t0=time.time()
    res['base'] = fit_q(Xf, yf, 0.5).predict(Xe)
    clf = xgb.XGBRegressor(objective='reg:logistic', **BASE)
    clf.fit(Xf, (yf > 0).astype(float))
    pc = clf.predict(Xe)
    pos = yf > 0
    Q = np.vstack([fit_q(Xf[pos], yf[pos], u).predict(Xe) for u in UP])  # 5 x n
    # corrected mixture median: u_pos = (0.5 - p0)/(1 - p0), clip [0.02, 0.5]
    us = np.clip((0.5 - pc) / (1 - pc), 0.02, 0.5)
    idx = np.clip((us - 0.1) / 0.1, 0, 4)  # knots at .1..0.5
    lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int); w = idx - lo
    mixmed = Q[lo, np.arange(len(pc))] * (1 - w) + Q[hi, np.arange(len(pc))] * w
    mixmed = np.where(pc >= 0.5, 0.0, mixmed)
    res['mixmed'] = mixmed
    res['b_base_mix'] = 0.5 * res['base'] + 0.5 * mixmed
    res['b75_base'] = 0.75 * res['base'] + 0.25 * mixmed
    auc = roc_auc_score(ye > 0, pc)
    print('fold<=%d %.0fs AUC=%.3f meanpc=%.3f actualzero=%.3f' % (fit_max, time.time()-t0, auc, pc.mean(), (ye==0).mean()))
    return res, pc, ye

r1, pc1, ye1 = run_fold(403, [431])
r2, pc2, ye2 = run_fold(375, [403, 431])
e431 = tr.snapshot_day == 431; e2 = tr.snapshot_day.isin([403, 431])
names = ['base','mixmed','b_base_mix','b75_base']
print('\n%-12s %9s %9s %9s' % ('variant','f1(431)','f2','avg'))
for nm in names:
    a = np.abs(r1[nm] - tr.future_spend_4w.values[e431]).mean()
    b = np.abs(r2[nm] - tr.future_spend_4w.values[e2]).mean()
    print('%-12s %9.2f %9.2f %9.2f' % (nm, a, b, (a+b)/2))
# calibration of classifier on fold1 eval
dfc = pd.DataFrame({'pc': pc1, 'y0': (ye1==0).astype(int)})
dfc['b'] = pd.qcut(dfc.pc, 10, duplicates='drop')
print(dfc.groupby('b', observed=True).agg(meanpc=('pc','mean'), zerorate=('y0','mean'), n=('y0','size')).round(3).to_string())
