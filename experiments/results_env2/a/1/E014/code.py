import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print("snapdays:", api.snapshot_days())
tt = api.train_targets()
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero_frac'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda x: (x==0).mean())
print(g.round(2))
d = g.reset_index()
sl, ic = np.polyfit(d.snapshot_day, d['mean'], 1)
print("trend slope/day %.4f intercept %.2f" % (sl, ic))
allF = api.load_saved('allF.parquet')
print("allF", allF.shape)
cols = list(allF.columns)
print("ncol", len(cols))
for i in range(0, len(cols), 12):
    print(i, cols[i:i+12])
m = tt.merge(allF, on=['household_key','snapshot_day'])
print("merged", m.shape)
corr = m.corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print("top |corr| with target:")
print(corr.head(30).round(3))
for name in ['e005_preds.parquet','e011_preds.parquet','e004_preds.parquet','repro_e5.parquet']:
    df = api.load_saved(name)
    print(name, df.shape)
    print(df.groupby('snapshot_day')['prediction'].agg(['count','mean','median']).round(2))
key = [c for c in cols if 'spend_28' in c.lower() or 'sp28' in c.lower() or 'spend_84' in c.lower()][:4]
print("key feats:", key)
if key:
    print(allF.groupby('snapshot_day')[key].mean().round(1))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
allF = api.load_saved('allF.parquet')
tt = api.train_targets()
m = tt.merge(allF.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'])
print("merged", m.shape)
corr = m.corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print("top |corr|:")
print(corr.head(35).round(3))
# per-snapshot means of key features vs target mean
keys = ['spend_28','spend_84','spend_364','spend_lag1y','spend_7','spend_seas_364','spend_seas_336','seas_ok','recency','zero28','wk_zero_12','wk_mean_12']
print(allF.groupby('snapshot_day')[keys].mean().round(1))
print("\ntrain target mean by snapshot:")
print(tt.groupby('snapshot_day')['future_spend_4w'].mean().round(1))
# distribution of target and of spend_28
print("\ntarget describe:"); print(tt['future_spend_4w'].describe().round(2))
print("\nspend_28 describe:"); print(allF['spend_28'].describe().round(2))
print("\nspend_84 describe:"); print(allF['spend_84'].describe().round(2))
# how well does raw spend_28 predict? MAE of spend_28 vs target on train
mm = m.dropna(subset=['spend_28'])
print("MAE spend_28 as pred:", np.abs(mm['spend_28']-mm['future_spend_4w']).mean().round(3))
print("MAE 0.9*spend_28:", np.abs(0.9*mm['spend_28']-mm['future_spend_4w']).mean().round(3))
print("MAE spend_84*0.31:", np.abs(0.31*mm['spend_84']-mm['future_spend_4w']).mean().round(3))
# check validation rows exist in allF
val = allF[allF.snapshot_day>=459]
print("\nval rows in allF:", val.shape, "unique hh:", val.household_key.nunique())
print(val.groupby('snapshot_day').size())


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
for name in ['e005_preds','e011_preds','e004_preds','e007_preds','e010_preds','e009_preds','e012_preds','repro_e5','e005_newfeats','lagfeats2']:
    try:
        df = api.load_saved(name+'.parquet')
        print(name, df.shape, list(df.columns)[:8])
        if 'prediction' in df.columns:
            print('   days:', sorted(df.snapshot_day.unique())[:20], 'pred mean %.2f median %.2f' % (df.prediction.mean(), df.prediction.median()))
    except Exception as e:
        print(name, 'ERR', type(e).__name__, e)


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float)
y = allF['future_spend_4w'].astype(float)
days = allF['snapshot_day'].astype(int)
d431 = (days==431)
y431 = y[d431].values
X431 = X[d431]
f = allF[d431]
print("431 rows:", d431.sum(), "target mean %.2f median %.2f" % (np.nanmean(y431), np.nanmedian(y431)))
for c,s in [('spend_28',1.0),('spend_28',0.9),('spend_84',0.31),('seq_mean',1.0),('wk_mean_12',1.0)]:
    print("baseline %s*%.2f MAE@431: %.3f" % (c,s,np.abs(s*f[c].values-y431).mean()))
def fit_pred(train_max, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, alpha=0.5):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    params = dict(objective='reg:quantileerror', quantile_alpha=alpha, tree_method='hist',
                  max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1)
    m = xgb.XGBRegressor(n_estimators=rounds, **params)
    t0=time.time()
    m.fit(X[tr], y[tr], sample_weight=w)
    p = m.predict(X431)
    mae = np.abs(p-y431).mean()
    print("snaps %d-%d rounds %d depth %d decay %d -> MAE@431 %.3f  predmean %.1f actmean %.1f  (%.0fs)" %
          (snaps_min, train_max, rounds, depth, decay, mae, p.mean(), np.nanmean(y431), time.time()-t0), flush=True)
    return p, mae
p0,m0 = fit_pred(403, 151)
p1,m1 = fit_pred(403, 207)
p2,m2 = fit_pred(403, 263)
for a in [0.3,0.5,0.7]:
    print("blend %.1f*model + %.1f*0.31*spend84: MAE@431 %.3f" % (a,1-a, np.abs(a*p0+(1-a)*0.31*f['spend_84'].values-y431).mean()))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
d431 = (days==431); y431 = y[d431].values; X431 = X[d431]; f = allF[d431]
def fit_pred(train_max=403, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, alpha=0.5, seed=1, ss=0.8, cs=0.8, mchild=1):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective='reg:quantileerror', quantile_alpha=alpha,
        tree_method='hist', max_depth=depth, learning_rate=lr, subsample=ss, colsample_bytree=cs,
        min_child_weight=mchild, nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m.predict(X431)
t0=time.time()
pA = fit_pred(depth=8, rounds=2400)
print("depth8: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pA-y431).mean(), pA.mean(), time.time()-t0), flush=True)
t0=time.time()
pB = fit_pred(depth=6, rounds=2400, decay=90)
print("decay90: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pB-y431).mean(), pB.mean(), time.time()-t0), flush=True)
t0=time.time()
pC = fit_pred(depth=6, rounds=2400, decay=220)
print("decay220: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pC-y431).mean(), pC.mean(), time.time()-t0), flush=True)
t0=time.time()
pD = fit_pred(depth=10, rounds=1600, lr=0.03)
print("depth10/1600: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pD-y431).mean(), pD.mean(), time.time()-t0), flush=True)
base = 0.31*f['spend_84'].values
for nm,p in [('A_d8',pA),('B_dc90',pB),('C_dc220',pC),('D_d10',pD)]:
    for a in [0.5,0.6,0.7]:
        print("%s blend a=%.1f: %.3f" % (nm,a,np.abs(a*p+(1-a)*base-y431).mean()))
# blend of models
pm = (pA+pB+pC+pD)/4
for a in [0.5,0.6,0.7]:
    print("model-avg blend a=%.1f: %.3f" % (a,np.abs(a*pm+(1-a)*base-y431).mean()))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(train_max, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, seed=1):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m
res = {}
t0=time.time()
for s in [319, 347, 375, 403]:
    te = (days==s); yt = y[te].values; Xt = X[te]; ft = allF[te]
    m = fit(s-28)
    p = m.predict(Xt)
    base = 0.31*ft['spend_84'].values
    seqm = ft['seq_mean'].values
    mae_p = np.abs(p-yt).mean()
    maes = {f'blend{a}': np.abs(a*p+(1-a)*base-yt).mean() for a in [0.5,0.6,0.7]}
    maes['seqonly'] = np.abs(seqm-yt).mean()
    maes['med3'] = np.median(np.vstack([p,base,seqm]),axis=0).astype(float)
    maes['med3'] = np.abs(maes['med3']-yt).mean()
    maes['mean3'] = np.abs((p+base+seqm)/3-yt).mean()
    print(s, "model %.3f" % mae_p, {k:round(v,3) for k,v in maes.items()}, flush=True)
print("total %.0fs" % (time.time()-t0))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = api.load_saved('allF.parquet')
tt = api.train_targets()
m = tt.merge(allF.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'])
# understand seas_ok
for s in [403, 431]:
    sub = m[m.snapshot_day==s]
    print(s, "n", len(sub), "seas_ok mean %.2f" % sub.seas_ok.mean(),
          "tenure>=364 frac %.2f" % (sub.tenure>=364).mean(),
          "spend_seas_364>0 frac %.2f" % (sub.spend_seas_364>0).mean(),
          "spend_lag1y>0 frac %.2f" % (sub.spend_lag1y>0).mean())
sub = m[(m.snapshot_day==431)]
s1 = sub[sub.seas_ok==1]
yt = s1['future_spend_4w'].values
print("\n@431 seas_ok==1 n=%d target mean %.1f" % (len(s1), yt.mean()))
for c in ['spend_seas_364','spend_seas_336','spend_lag1y','spend_84','spend_28','seq_mean']:
    print("  %s: corr %.3f MAE %.2f" % (c, np.corrcoef(s1[c],yt)[0,1], np.abs(s1[c].values-yt).mean()))
print("  0.31*spend_84 MAE %.2f" % np.abs(0.31*s1['spend_84'].values-yt).mean())
print("  0.5*(seas364+seas336) MAE %.2f" % np.abs(0.5*(s1['spend_seas_364']+s1['spend_seas_336']).values-yt).mean())
print("  mean(seas364,seas336,0.31*84) MAE %.2f" % np.abs(((s1['spend_seas_364']+s1['spend_seas_336'])/2*0.5+0.31*s1['spend_84']*0.5).values-yt).mean())
# and on the seas_ok==0 subset, baseline behavior
s0 = sub[sub.seas_ok==0]
yt0 = s0['future_spend_4w'].values
print("@431 seas_ok==0 n=%d: 0.31*84 MAE %.2f, spend_28 MAE %.2f" % (len(s0), np.abs(0.31*s0['spend_84'].values-yt0).mean(), np.abs(s0['spend_28'].values-yt0).mean()))
# tenure distribution overall
print("\ntenure describe:"); print(allF['tenure'].describe().round(0))
print("frac tenure>=364 by snapshot:")
print(allF.groupby('snapshot_day')['tenure'].apply(lambda x:(x>=364).mean()).round(2))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
d431=(days==431); y431=y[d431].values; X431=X[d431]; f=allF[d431]
tr=(days>=151)&(days<=403)&y.notna()
w=0.5**(((403)-days[tr].values)/140)
m=xgb.XGBRegressor(n_estimators=2400, objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
    max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=1)
m.fit(X[tr],y[tr],sample_weight=w)
p=m.predict(X431)
# zero structure
z28 = f['spend_28'].values==0
z84 = f['spend_84'].values==0
print("frac spend_28==0: %.3f, actual target==0 among them: %.3f" % (z28.mean(), (y431[z28]==0).mean()))
print("frac spend_84==0: %.3f, actual target==0 among them: %.3f" % (z84.mean(), (y431[z84]==0).mean()))
print("model pred for spend_28==0: mean %.2f median %.2f, actual mean %.2f" % (p[z28].mean(), np.median(p[z28]), y431[z28].mean()))
print("MAE on z28 subset: model %.2f, zero-pred %.2f, actual-zero-frac %.2f" % (np.abs(p[z28]-y431[z28]).mean(), np.abs(y431[z28]).mean(), (y431[z28]==0).mean()))
# clipping thresholds
base=0.31*f['spend_84'].values
for thr in [0,5,10,15,20,30]:
    pc = p.copy(); pc[p<thr]=0
    print("clip thr %d: model MAE %.3f, blend MAE %.3f" % (thr, np.abs(pc-y431).mean(), np.abs(0.5*pc+0.5*base-y431).mean()))
# blend refinements
seqm=f['seq_mean'].values; s28=f['spend_28'].values
cands = {
 '0.5p+0.5b': 0.5*p+0.5*base,
 '0.45p+0.45b+0.1seq': 0.45*p+0.45*base+0.1*seqm,
 '0.5p+0.4b+0.1s28': 0.5*p+0.4*base+0.1*s28,
 '0.55p+0.35b+0.1s28': 0.55*p+0.35*base+0.1*s28,
 '0.5p+0.5b clip10': np.where(0.5*p+0.5*base<10,0,0.5*p+0.5*base),
 '0.5p+0.5b clip15': np.where(0.5*p+0.5*base<15,0,0.5*p+0.5*base),
 '0.5p+0.5b clip20': np.where(0.5*p+0.5*base<20,0,0.5*p+0.5*base),
}
for k,v in cands.items():
    print("%-22s MAE %.3f" % (k, np.abs(v-y431).mean()))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(te_day, obj='reg:quantileerror', alpha=0.5, rounds=2400, lr=0.03, depth=6, decay=140, seed=1, logt=False):
    train_max = te_day-28
    tr = (days>=151)&(days<=train_max)&y.notna()
    w = 0.5**((train_max-days[tr].values)/decay)
    yy = np.log1p(y[tr]) if logt else y[tr]
    m = xgb.XGBRegressor(n_estimators=rounds, objective=obj, quantile_alpha=alpha, tree_method='hist',
        max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=seed)
    m.fit(X[tr], yy, sample_weight=w)
    return m
def evaluate(te_day, m, logt=False):
    te=(days==te_day); yt=y[te].values; ft=allF[te]
    p=m.predict(X[te])
    if logt: p=np.expm1(p)
    base=0.31*ft['spend_84'].values
    out={'model':np.abs(p-yt).mean()}
    for a in [0.5,0.55,0.6]:
        b=a*p+(1-a)*base
        out[f'blend{a}']=np.abs(b-yt).mean()
        bc=np.where(b<10,0,b)
        out[f'blend{a}clip']=np.abs(bc-yt).mean()
    return out
t0=time.time()
# squared-error model at 431
msq = fit(431, obj='reg:squarederror')
print("431 sq:", {k:round(v,3) for k,v in evaluate(431, msq).items()}, flush=True)
# log-target squared model at 431
mlog = fit(431, obj='reg:squarederror', logt=True)
print("431 log:", {k:round(v,3) for k,v in evaluate(431, mlog, logt=True).items()}, flush=True)
# quantile alpha 0.45 at 431
mq45 = fit(431, alpha=0.45)
print("431 q45:", {k:round(v,3) for k,v in evaluate(431, mq45).items()}, flush=True)
print("%.0fs" % (time.time()-t0))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(train_max, obj='reg:quantileerror', alpha=0.5, rounds=2400, lr=0.03, depth=6, decay=140, seed=1):
    tr = (days>=151)&(days<=train_max)&y.notna()
    w = 0.5**((train_max-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective=obj, quantile_alpha=alpha, tree_method='hist',
        max_depth=6, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m
def evaluate(te_day, m, tag=''):
    te=(days==te_day); yt=y[te].values; ft=allF[te]
    p=m.predict(X[te]); base=0.31*ft['spend_84'].values
    out={'model':np.abs(p-yt).mean()}
    for a in [0.5,0.55,0.6]:
        b=a*p+(1-a)*base
        out[f'b{a}']=np.abs(b-yt).mean()
        out[f'b{a}c']=np.abs(np.where(b<10,0,b)-yt).mean()
    print(te_day, tag, {k:round(v,2) for k,v in out.items()}, flush=True)
    return out
t0=time.time()
for s in [347, 375, 403]:
    evaluate(s, fit(s-28), 'q50')
print("%.0fs" % (time.time()-t0))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
vmask = days>=459
Xv = X[vmask]; val = allF[vmask]
tr = (days>=151)&(days<=403)&y.notna()
w = 0.5**((403-days[tr].values)/140)
preds=[]; t0=time.time()
for seed in [1,7,42]:
    m = xgb.XGBRegressor(n_estimators=2400, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    preds.append(m.predict(Xv))
    print("seed", seed, "done %.0fs" % (time.time()-t0), flush=True)
p = np.mean(preds, axis=0)
base = 0.31*val['spend_84'].values
b = 0.5*p + 0.5*base
b = np.where(b<10, 0, b)
out = pd.DataFrame({'household_key': val['household_key'].values,
                    'snapshot_day': val['snapshot_day'].values,
                    'prediction': b})
print(out.shape)
print(out.groupby('snapshot_day')['prediction'].agg(['count','mean','median']).round(2))
print("pred mean overall %.2f (E005 was 127.10)" % out.prediction.mean())
path = api.save_table(out, 'e014_preds.parquet')
print("saved:", path)
