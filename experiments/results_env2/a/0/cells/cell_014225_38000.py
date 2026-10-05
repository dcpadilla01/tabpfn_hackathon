
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets(); train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')
tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
BASE = dict(learning_rate=0.08, n_estimators=400, subsample=0.9, colsample_bytree=0.9,
            tree_method='hist', n_jobs=8)
CFGS = [(4,20),(4,40),(5,40),(6,40)]
UP = np.array([0.1,0.2,0.3,0.4,0.5])
def mko(u, sw=None, obj='reg:quantileerror'):
    return xgb.XGBRegressor(objective=obj, quantile_alpha=u if obj=='reg:quantileerror' else 0.5, **BASE)

def run_fold(fit_max, eval_days, tau=None):
    fi = tr.snapshot_day <= fit_max; ei = tr.snapshot_day.isin(eval_days)
    Xf = tr.loc[fi, feat_cols].values.astype(np.float32); yf = tr.future_spend_4w.values[fi]
    Xe = tr.loc[ei, feat_cols].values.astype(np.float32); ye = tr.future_spend_4w.values[ei]
    e4w = tr.loc[ei,'exp4w_blend'].values
    sd = tr.snapshot_day.values[fi]
    sw = np.exp((sd-fit_max)/tau) if tau else None
    t0=time.time()
    pb = np.mean([mko(0.5, sw).fit(Xf,yf,sample_weight=sw).predict(Xe) for d,mw in CFGS
                  for m in [xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=d, min_child_weight=mw, **BASE)] ], axis=0)
    clf = xgb.XGBRegressor(objective='reg:logistic', max_depth=5, min_child_weight=40, **BASE)
    clf.fit(Xf,(yf>0).astype(float), sample_weight=sw); pc = clf.predict(Xe)
    pos = yf>0; Xfp = Xf[pos]; yfp = yf[pos]; swp = sw[pos] if tau else None
    Q = []
    for u in UP:
        Q.append(np.mean([xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, max_depth=d, min_child_weight=mw, **BASE
                                           ).fit(Xfp,yfp,sample_weight=swp).predict(Xe) for d,mw in CFGS], axis=0))
    Q = np.vstack(Q)
    us = np.clip((pc-0.5)/np.maximum(pc,1e-6), 0.02, 0.5)
    idx = np.clip((us-UP[0])/(UP[1]-UP[0]), 0, len(UP)-1)
    lo=np.floor(idx).astype(int); hi=np.ceil(idx).astype(int); w=idx-lo
    mm = Q[lo,np.arange(len(pc))]*(1-w) + Q[hi,np.arange(len(pc))]*w
    mm = np.where(pc<=0.5, 0.0, mm)
    out = {'base':pb, 'mix50':0.5*pb+0.5*mm, 'mix25':0.75*pb+0.25*mm}
    for nm, p in [('base',pb),('mix50',out['mix50'])]:
        for wt in [0.6,0.7,0.8,0.9,1.0]:
            out['%s_w%d'%(nm,int(wt*10))] = wt*p + (1-wt)*e4w
    print('fold<=%d tau=%s %.0fs' % (fit_max, tau, time.time()-t0))
    return out, ye

res = {}
for tau in [None, 250]:
    r1, y1 = run_fold(403, [431], tau)
    r2, y2 = run_fold(375, [403,431], tau)
    for nm in r1:
        a = np.abs(r1[nm]-y1).mean(); b = np.abs(r2[nm]-y2).mean()
        res[(nm,tau)] = (a+b)/2
w1 = pd.DataFrame({nm:v for (nm,t),v in res.items() if t is None}, index=['MAE']).T.sort_values('MAE')
w2 = pd.DataFrame({nm:v for (nm,t),v in res.items() if t==250}, index=['MAE']).T.sort_values('MAE')
print('\ntau=None:'); print(w1.round(2).to_string())
print('\ntau=250:'); print(w2.round(2).to_string())
