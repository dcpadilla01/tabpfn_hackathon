import agent_api as api, pandas as pd, numpy as np
names = ["allF","oof_e5","repro_e5","e003_preds","e004_preds","e005_preds","e007_preds","e009_preds","e010_preds","e011_preds","e012_preds","e014_preds","e016_preds","e017_preds","e005_newfeats","e004_new","lagfeats","lagfeats2","f_weekly","camp_feats","e016_allpreds","e016_held","e016_sqpreds","e002_features"]
for nm in names:
    try:
        df = api.load_saved(nm + ".parquet")
        days = sorted(df['snapshot_day'].unique()) if 'snapshot_day' in df.columns else None
        print(f"{nm}: shape={df.shape} days={days} cols={list(df.columns)[:12]}")
    except Exception as e:
        print(f"{nm}: ERR {type(e).__name__} {e}")


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
tt = api.train_targets()
print("targets:", tt.shape, tt.columns.tolist())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1))
allF = api.load_saved('allF.parquet')
print("allF:", allF.shape)
print("cols:", list(allF.columns))
print(allF.groupby('snapshot_day').size())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
oof = api.load_saved('oof_e5.parquet')
oof['err'] = oof['pred'] - oof['future_spend_4w']
g = oof.groupby('snapshot_day').agg(n=('err','size'), mae=('err', lambda x: np.abs(x).mean()), bias=('err','mean'), act=('future_spend_4w','mean'), pred=('pred','mean'))
print(g.round(2))
print("\nOverall OOF MAE:", np.abs(oof['err']).mean().round(3))
# fit linear bias ~ day on snapshots <=347, apply to 375
tr = oof[oof.snapshot_day<=347]
A = np.vstack([tr.snapshot_day, np.ones(len(tr))]).T
coef, *_ = np.linalg.lstsq(A, tr['err'].values, rcond=None)
print("bias trend coef (per day, intercept):", coef.round(4))
te = oof[oof.snapshot_day==375]
corr_pred = te['pred'] - (coef[0]*375 + coef[1])
print("375 MAE raw:", np.abs(te['err']).mean().round(3), " corrected:", np.abs(corr_pred-te['future_spend_4w']).mean().round(3))
# also quadratic
A2 = np.vstack([tr.snapshot_day**2, tr.snapshot_day, np.ones(len(tr))]).T
c2, *_ = np.linalg.lstsq(A2, tr['err'].values, rcond=None)
corr2 = te['pred'] - (c2[0]*375**2 + c2[1]*375 + c2[2])
print("375 MAE quad-corrected:", np.abs(corr2-te['future_spend_4w']).mean().round(3))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
# Check e016_allpreds columns on train rows: are they OOF? compare to target
ap = api.load_saved('e016_allpreds.parquet')
tt = api.train_targets()
m = ap.merge(tt, on=['household_key','snapshot_day'])
print(m.shape)
for c in ['pq','pl','pc','pa']:
    print(c, "MAE:", np.abs(m[c]-m['future_spend_4w']).mean().round(3), "bias:", (m[c]-m['future_spend_4w']).mean().round(3))
# blend of pq..pa on train rows
m['blend'] = m[['pq','pl','pc','pa']].mean(axis=1)
print("mean blend MAE:", np.abs(m['blend']-m['future_spend_4w']).mean().round(3))
sq = api.load_saved('e016_sqpreds.parquet')
m2 = sq.merge(tt, on=['household_key','snapshot_day'])
for c in ['sq0','sq1','sq2']:
    print(c, "MAE:", np.abs(m2[c]-m2['future_spend_4w']).mean().round(3))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
oof = api.load_saved('oof_e5.parquet')
te = oof[oof.snapshot_day==375].copy()
tr = oof[oof.snapshot_day<=347].copy()
res = tr['future_spend_4w'] - tr['pred']
# MAE-optimal constant shift = median of residuals
sh = np.median(res)
print("median residual (const shift):", round(sh,2))
print("375 raw MAE:", round(np.abs(te['err']).mean(),3))
print("375 const-shift MAE:", round(np.abs(te['pred']+sh-te['future_spend_4w']).mean(),3))
# multiplicative: pred*k + b, optimize on tr via grid on k, median for b
best=None
for k in np.arange(0.9,1.6,0.02):
    r = tr['future_spend_4w'] - k*tr['pred']
    b = np.median(r)
    mae_tr = np.abs(k*tr['pred']+b-tr['future_spend_4w']).mean()
    if best is None or mae_tr<best[0]: best=(mae_tr,k,b)
print("best affine on tr:", best)
k,b = best[1],best[2]
print("375 affine MAE:", round(np.abs(k*te['pred']+b-te['future_spend_4w']).mean(),3))
# per-snapshot-day shift from OOF of that same day would be cheating; but shift from PREVIOUS day only:
# use shift estimated on snapshot 347 OOF alone, apply to 375
r347 = oof[oof.snapshot_day==347]
sh347 = np.median(r347['future_spend_4w']-r347['pred'])
print("shift from 347 only:", round(sh347,2), "-> 375 MAE:", round(np.abs(te['pred']+sh347-te['future_spend_4w']).mean(),3))
# avg of last 3 days' shifts
shs = [np.median(oof[oof.snapshot_day==d]['future_spend_4w']-oof[oof.snapshot_day==d]['pred']) for d in [291,319,347]]
print("shifts:", [round(s,1) for s in shs], "avg:", round(np.mean(shs),2), "-> 375 MAE:", round(np.abs(te['pred']+np.mean(shs)-te['future_spend_4w']).mean(),3))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
held = api.load_saved('e016_held.parquet')
print("e016_held (403/431 held-out from sq training):")
for c in ['sq0','sq1','sq2']:
    print(c, "MAE:", np.abs(held[c]-held['future_spend_4w']).mean().round(3), "bias:", (held[c]-held['future_spend_4w']).mean().round(2))
# compare with oof_e5 at same days
oof = api.load_saved('oof_e5.parquet')
for d in [403,431]:
    sub = held[held.snapshot_day==d]
    print(d, "sq MAE:", np.abs(sub['sq1']-sub['future_spend_4w']).mean().round(2), "| oof_e5 MAE at", d, ":", end=" ")
    o = oof[oof.snapshot_day==d]
    if len(o): print(round(np.abs(o['pred']-o['future_spend_4w']).mean(),2))
    else: print("n/a")
# validation pred distributions
for nm in ['e005_preds','e011_preds','e016_preds','e017_preds']:
    p = api.load_saved(nm+'.parquet')
    print(nm, "val pred mean/median:", p['prediction'].mean().round(1), p['prediction'].median().round(1))
oofv = oof[oof.snapshot_day==375]
print("oof 375 pred mean:", oofv['pred'].mean().round(1), "actual mean:", oofv['future_spend_4w'].mean().round(1))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
tt = api.train_targets()
print("Per-snapshot-day intrinsic difficulty (MAE of predicting the day's median):")
for d, g in tt.groupby('snapshot_day'):
    med = g['future_spend_4w'].median()
    print(f"day {d}: n={len(g)} mean={g['future_spend_4w'].mean():6.1f} std={g['future_spend_4w'].std():6.1f} medianMAE={np.abs(g['future_spend_4w']-med).mean():6.1f} zero_share={(g['future_spend_4w']==0).mean():.3f}")


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
allF = api.load_saved('allF.parquet')
feats = ['spend_28','spend_84','spend_364','actdays_28','recency','bask_28','spend_total','tenure']
print("Feature means by snapshot day (drift check):")
print(allF.groupby('snapshot_day')[feats].mean().round(1))
# household overlap train vs val
tr_h = set(allF[allF.snapshot_day<=431].household_key); va_h = set(allF[allF.snapshot_day>=459].household_key)
print("train hh:", len(tr_h), "val hh:", len(va_h), "val not in train:", len(va_h-tr_h))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
DROP = ['household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in allF.columns if c not in DROP]
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
print("tr:", tr.shape, "te:", te.shape, "n_feats:", len(FEATS))

def w(day, halflife=140, ref=375): return 0.5**((ref-day)/halflife)

def run(model_type, alpha=0.5, tweedie_p=1.3, lr=0.03, rounds=1200, seeds=(1,2), logt=False):
    t0=time.time(); preds=[]
    Xtr, ytr = tr[FEATS], tr['future_spend_4w']
    wt = w(tr['snapshot_day'].values)
    yt = np.log1p(ytr) if logt else ytr
    for s in seeds:
        if model_type=='quantile':
            m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        elif model_type=='tweedie':
            m = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=tweedie_p, n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        else:
            m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        m.fit(Xtr, yt, sample_weight=wt)
        p = m.predict(te[FEATS])
        if logt: p = np.expm1(p)
        preds.append(np.clip(p,0,None))
    P = np.mean(preds,axis=0)
    mae = np.abs(P-te['future_spend_4w']).values.mean()
    print(f"{model_type} a={alpha} tp={tweedie_p} logt={logt} r={rounds}: honest403/431 MAE={mae:.3f} bias={(P-te['future_spend_4w'].values).mean():.1f} ({time.time()-t0:.0f}s)")
    return P

Pq = run('quantile')
Pl = run('squared', logt=True)
Pt = run('tweedie', tweedie_p=1.3)
Pt2 = run('tweedie', tweedie_p=1.1)
print("blend q+logl:", np.abs(0.5*Pq+0.5*Pl-te['future_spend_4w'].values).mean().round(3))
print("blend q+tweedie1.3:", np.abs(0.5*Pq+0.5*Pt-te['future_spend_4w'].values).mean().round(3))
print("blend q+logl+t1.3+t1.1:", np.abs((Pq+Pl+Pt+Pt2)/4-te['future_spend_4w'].values).mean().round(3))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def run(FEATS, lr=0.03, rounds=1200, hl=140, depth=6, mcw=25, ss=0.7, cs=0.7, seeds=(1,2), ref=375, tag=""):
    t0=time.time(); preds=[]
    wt = 0.5**((ref-tr['snapshot_day'].values)/hl)
    for s in seeds:
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=rounds, learning_rate=lr,
                             max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=s, tree_method='hist')
        m.fit(tr[FEATS], ytr, sample_weight=wt)
        preds.append(np.clip(m.predict(te[FEATS]),0,None))
    P = np.mean(preds,axis=0)
    mae = np.abs(P-yte).mean()
    print(f"{tag:34s} honest MAE={mae:.3f} bias={(P-yte).mean():6.1f} ({time.time()-t0:.0f}s)")
    return P, mae

_, m0 = run(BASE, tag="base (E011 recipe)")
F2 = BASE + ['snapshot_day']
_, m1 = run(F2, tag="+snapshot_day feature")
_, m2 = run(BASE, rounds=2400, lr=0.02, tag="2400r lr.02")
_, m3 = run(BASE, hl=277, tag="hl=277")
_, m4 = run(BASE, hl=70, tag="hl=70")
_, m5 = run(BASE, hl=100000, tag="no decay")
_, m6 = run(BASE, depth=5, tag="depth5")
_, m7 = run(BASE, depth=8, mcw=50, tag="depth8 mcw50")
_, m8 = run(BASE, ss=0.9, cs=0.9, tag="ss.9 cs.9")
print("\nbest so far:", min([(m0,'base'),(m1,'+day'),(m2,'2400r'),(m3,'hl277'),(m4,'hl70'),(m5,'nodecay'),(m6,'d5'),(m7,'d8'),(m8,'ss9')]))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def train(FEATS, lr=0.03, rounds=1200, hl=None, depth=6, mcw=25, ss=0.7, cs=0.7, seed=1, ref=375):
    wt = np.ones(len(tr)) if hl is None else 0.5**((ref-tr['snapshot_day'].values)/hl)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS], ytr, sample_weight=wt)
    return m

# 1) bias by prediction bucket (base config, seed 1)
m = train(BASE)
P = np.clip(m.predict(te[BASE]),0,None)
b = pd.qcut(P, 8, duplicates='drop')
g = pd.DataFrame({'p':P,'y':yte,'b':b}).groupby('b',observed=True).agg(n=('y','size'),pred=('p','mean'),act=('y','mean'),mae=('p',lambda x: 0))
g['mae'] = pd.DataFrame({'p':P,'y':yte,'b':b}).groupby('b',observed=True).apply(lambda x: np.abs(x['p']-x['y']).mean(), include_groups=False)
print("Bias by prediction octile (honest 403/431):"); print(g.round(1))

# 2) feature importance screening: train once, keep top-k
imp = pd.Series(m.feature_importances_, index=BASE).sort_values(ascending=False)
print("\ntop15:", list(imp.head(15).round(3).items()))
for k in [40, 70]:
    Fk = list(imp.head(k).index)
    preds=[]
    for s in (1,2):
        mm = train(Fk, seed=s)
        preds.append(np.clip(mm.predict(te[Fk]),0,None))
    Pk = np.mean(preds,axis=0)
    print(f"top{k} feats: honest MAE={np.abs(Pk-yte).mean():.3f}")
# 3) combined small winners: no decay + snapshot_day + hl277 ensemble
F3 = BASE + ['snapshot_day']
preds=[]
for cfg in [dict(hl=None), dict(hl=277), dict(hl=None), dict(hl=277)]:
    for s in (1,2):
        mm = train(F3, seed=s, **cfg)
        preds.append(np.clip(mm.predict(te[F3]),0,None))
P3 = np.mean(preds,axis=0)
print(f"diverse(no-decay+hl277, +day): honest MAE={np.abs(P3-yte).mean():.3f}")


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def train(FEATS, alpha=0.5, lr=0.03, rounds=1200, hl=None, depth=6, mcw=25, ss=0.7, cs=0.7, seed=1, ref=375):
    wt = np.ones(len(tr)) if hl is None else 0.5**((ref-tr['snapshot_day'].values)/hl)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS], ytr, sample_weight=wt)
    return m

# base honest preds (2 seeds, no decay)
Ps = []
for s in (1,2):
    m = train(BASE, seed=s); Ps.append(np.clip(m.predict(te[BASE]),0,None))
P = np.mean(Ps,axis=0)
res = yte - P
print("honest bias:", res.mean().round(2), "median resid:", np.median(res).round(2))
print("raw MAE:", np.abs(res).mean().round(3))
for c in [5,10,15,20,np.median(res)]:
    print(f"shift +{c:.1f}: MAE={np.abs(P+c-yte).mean():.3f}")
# alpha variants
for a in [0.52, 0.55, 0.58]:
    Ps=[]
    for s in (1,2):
        m = train(BASE, alpha=a, seed=s); Ps.append(np.clip(m.predict(te[BASE]),0,None))
    Pa = np.mean(Ps,axis=0)
    print(f"alpha={a}: MAE={np.abs(Pa-yte).mean():.3f} bias={(Pa-yte).mean():.1f}")
# top70 feats + no decay (best pruning) with shift
imp = pd.Series(train(BASE).feature_importances_, index=BASE).sort_values(ascending=False)
F70 = list(imp.head(70).index)
Ps=[]
for s in (1,2):
    m = train(F70, seed=s); Ps.append(np.clip(m.predict(te[F70]),0,None))
P70 = np.mean(Ps,axis=0)
print("top70 nodecay raw:", np.abs(P70-yte).mean().round(3), "| +10:", np.abs(P70+10-yte).mean().round(3), "| +15:", np.abs(P70+15-yte).mean().round(3))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
DROP = ['household_key','snapshot_day','future_spend_4w']
BASE = [c for c in allF.columns if c not in DROP]
tr = allF[allF.snapshot_day<=375]
va = allF[allF.snapshot_day>=459]
ytr = tr['future_spend_4w'].values
print("train rows:", len(tr), "val rows:", len(va))

def fit(F, alpha, seed, rounds=1200, lr=0.03, depth=6, mcw=25, ss=0.7, cs=0.7):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs,
                         n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[F], ytr)  # uniform weights: no time decay
    return m

# feature pruning from a seed-1 alpha=.5 model
m0 = fit(BASE, 0.5, 1)
imp = pd.Series(m0.feature_importances_, index=BASE).sort_values(ascending=False)
F70 = list(imp.head(70).index)

t0=time.time()
preds=[]
for s in range(1,9):
    m = fit(F70, 0.52, s)
    preds.append(np.clip(m.predict(va[F70]),0,None))
P = np.mean(preds,axis=0)
print(f"8 seeds done ({time.time()-t0:.0f}s)  val pred mean={P.mean():.1f} median={np.median(P):.1f} min={P.min():.2f} max={P.max():.1f} finite={np.isfinite(P).all()}")

out = va[['household_key','snapshot_day']].copy()
out['prediction'] = P
path = api.save_table(out, 'e018_preds.parquet')
print("saved:", path, out.shape)
