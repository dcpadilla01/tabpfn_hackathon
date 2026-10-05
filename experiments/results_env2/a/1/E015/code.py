
import agent_api, numpy as np, pandas as pd

tt = agent_api.train_targets()
print("train rows:", tt.shape)
print(tt['future_spend_4w'].describe())
print("quantiles:", np.percentile(tt['future_spend_4w'], [50,75,90,95,99]))

# load saved prediction tables and check correlation / per-day bias on train rows
names = ["e005_preds","e004_preds","e003_preds","e011_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds","e001_preds","e002_preds"]
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
        p = m['prediction']
        t = m['future_spend_4w']
        print(n, "cols:", list(df.columns), "mae:", np.mean(np.abs(p-t)).round(3),
              "bias:", np.mean(p-t).round(3), "corr:", np.corrcoef(p,t)[0,1].round(3))
    except Exception as e:
        print(n, "ERR", type(e).__name__, e)


# ---- cell ----

import agent_api, numpy as np, pandas as pd
tt = agent_api.train_targets()
df = agent_api.load_saved("e005_preds.parquet")
print(df.shape, list(df.columns))
print(df.head())
m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
print(m.shape, list(m.columns))
print(m.head())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
print("allF:", allF.shape)
print(list(allF.columns))
print(allF['snapshot_day'].value_counts().sort_index())
# check other saved tables
for n in ["e005_newfeats","lagfeats","lagfeats2","f_weekly","camp_feats","repro_e5"]:
    try:
        d = agent_api.load_saved(n + ".parquet")
        print(n, d.shape, list(d.columns)[:12])
    except Exception as e:
        print(n, "ERR", e)


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
# per-day bias of e005 preds on train? we only have val preds for e005. Use repro_e5 (p13,p11) also val-only.
# Instead: train-side analysis. Correlation of features with target, and per-day target stats.
g = m.groupby("snapshot_day")['future_spend_4w'].agg(['mean','median','count'])
print(g.round(2))
# overall: how much of target is 0?
print("zero share train:", (m.future_spend_4w==0).mean().round(4))
# weekly seasonality: is target related to week_of_year?
m['wk'] = ((m.snapshot_day+8)//7) % 52
print(m.groupby('wk')['future_spend_4w'].mean().round(1).to_string())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], how="inner", suffixes=('_F','_T'))
print("merged:", m.shape)
print(m[['future_spend_4w_F','future_spend_4w_T']].head())
print("equal:", np.allclose(m.future_spend_4w_F, m.future_spend_4w_T))
t = m['future_spend_4w_T']
g = m.groupby("snapshot_day").apply(lambda d: pd.Series({'mean':d.future_spend_4w_T.mean(),'med':d.future_spend_4w_T.median(),'zero':(d.future_spend_4w_T==0).mean()}))
print(g.round(2))
m['wk'] = ((m.snapshot_day+8)//7) % 52
print(m.groupby('wk')['future_spend_4w_T'].mean().round(1).to_string())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
v = allF[allF.snapshot_day>=459]
print("val rows in allF:", v.shape)
print(v[['household_key','snapshot_day','future_spend_4w']].head(10))
print("nan count val:", v['future_spend_4w'].isna().sum())
tr = allF[allF.snapshot_day<459]
print("nan count train:", tr['future_spend_4w'].isna().sum())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
# Look at validation rows: what does the actual future spend look like for val snapshots?
# We can't see beyond 459 via snapshot/history caps, but build_features sees up to snapshot day.
# Key question: is there a systematic per-day effect (e.g. week-of-year seasonality) we can exploit?
# Check: at train snapshots, target mean by snapshot day vs week_of_year of the FUTURE window.
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"])
# future window week: weeks (day+1..day+28) -> mean week index
m['fut_wk'] = ((m.snapshot_day + 15) + 8) // 7 % 52
g = m.groupby('fut_wk')['future_spend_4w'].agg(['mean','count'])
print(g.round(1).to_string())
# Also check spend_28 mean by future week at train (to see if level shift is household-driven)


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
m['t'] = m.future_spend_4w_T
m['fut_wk'] = ((m.snapshot_day + 15) + 8) // 7 % 52
g = m.groupby('fut_wk')['t'].agg(['mean','count'])
print(g.round(1).to_string())
# spend_28 mean by fut_wk (household mix constant-ish)
g2 = m.groupby('fut_wk')['spend_28'].mean()
print(g2.round(1).to_string())
# ratio target/spend_28 by fut_wk
print((g['mean']/g2).round(3).to_string())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
names = ["e005_preds","e011_preds","e004_preds","e003_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds"]
preds = {}
for n in names:
    d = agent_api.load_saved(n + ".parquet")
    preds[n] = d.set_index(["household_key","snapshot_day"])['prediction']
P = pd.DataFrame(preds)
print(P.shape)
C = P.corr()
print(C.round(4).to_string())
print("\nmean pred per table:", P.mean().round(2).to_string())
print("\nstd per table:", P.std().round(2).to_string())
# equal-weight blend of the three best
b3 = P[['e005_preds','e011_preds','e004_preds']].mean(axis=1)
print("\nblend3 vs e005: corr", np.corrcoef(b3, P['e005_preds'])[0,1].round(5), "mean", b3.mean().round(2))


# ---- cell ----

import agent_api, numpy as np, pandas as pd
P = {}
for n in ["e005_preds","e011_preds","e004_preds","e003_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds"]:
    P[n] = agent_api.load_saved(n + ".parquet").set_index(["household_key","snapshot_day"])['prediction']
P = pd.DataFrame(P)
# per snapshot-day mean prediction, val days
v = P.reset_index()
g = v.groupby('snapshot_day').agg(e005=('e005_preds','mean'), e011=('e011_preds','mean'),
                                  e004=('e004_preds','mean'), e003=('e003_preds','mean'),
                                  e008=('e008_preds','mean'), e009=('e009_preds','mean'))
print(g.round(2).to_string())
# what does e005 say vs e008 (trained incl 431)? e008 mean 139.76 vs e005 127.10
# Train-side target means by day: 403:146.5, 431:141.0. Later train days higher -> e008 higher preds.
# Check e005's per-day MAE pattern can't be computed (no val targets). Use train-side surrogate:
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
print("\ntrain target mean by day:\n", m.groupby('snapshot_day')['future_spend_4w_T'].mean().round(1).to_string())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
tr = m[m.snapshot_day<=431].copy()
# candidate: week-of-year effect on target beyond household features?
# Fit quick surrogate: residual of a simple model vs fut_wk
import sklearn.ensemble as ske
feats = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w_F','future_spend_4w_T','t','fut_wk')]
X = tr[feats].fillna(-1).values
y = tr.future_spend_4w_T.values
# simple: use spend_28 as base predictor, residual analysis
r = y - tr.spend_28.values
tr['fut_wk'] = ((tr.snapshot_day+15+8)//7) % 52
print(tr.groupby('fut_wk')['r'].agg(['mean','count']).round(2).to_string())
# residual by snapshot day
print("\nresid mean by snapshot_day:\n", tr.groupby('snapshot_day')['r'].mean().round(2).to_string())


# ---- cell ----

import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
tr = m[m.snapshot_day<=431].copy()
tr['r'] = tr.future_spend_4w_T - tr.spend_28
tr['fut_wk'] = ((tr.snapshot_day+15+8)//7) % 52
print(tr.groupby('fut_wk')['r'].agg(['mean','count']).round(2).to_string())
print("\nresid mean by snapshot_day:\n", tr.groupby('snapshot_day')['r'].mean().round(2).to_string())
# trend in target over snapshot day? fit linear
d = tr.groupby('snapshot_day')['r'].mean()
print("\nslope of resid vs day:", np.polyfit(d.index, d.values, 1)[0].round(4))


# ---- cell ----

import agent_api, numpy as np, pandas as pd, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")

allF = agent_api.load_saved("allF.parquet")
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr_days = [123,151,179,207,235,263,291,319,347,375]
tr = allF[allF.snapshot_day.isin(tr_days)]
va = allF[allF.snapshot_day==403]
Xtr, ytr = tr[feats].values, tr.future_spend_4w.values
Xva, yva = va[feats].values, va.future_spend_4w.values
w = 0.5 ** ((375 - tr.snapshot_day.values)/140.0)
print("train", Xtr.shape, "val", Xva.shape)

def fit_q(yt, seed, depth=6):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=0.03, n_estimators=1200, max_depth=depth, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)

t0=time.time()
base = np.maximum(Xva[:, feats.index('spend_28')], 0)
print("baseline spend_28 MAE:", np.abs(base-yva).mean().round(3), f"({time.time()-t0:.0f}s)")

res = {}
t0=time.time(); pA = np.mean([fit_q(ytr, s) for s in (1,2,3)], axis=0); res['A_lin_d6']=pA
print("A lin d6 MAE:", np.abs(pA-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pB = np.mean([np.expm1(np.maximum(fit_q(np.log1p(ytr), s),0)) for s in (1,2,3)], axis=0); res['B_log_d6']=pB
print("B log d6 MAE:", np.abs(pB-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pC = np.mean([np.maximum(fit_q(np.sqrt(ytr), s),0)**2 for s in (1,2,3)], axis=0); res['C_sqrt_d6']=pC
print("C sqrt d6 MAE:", np.abs(pC-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pD = np.mean([fit_q(ytr, s, depth=4) for s in (1,2,3)], axis=0); res['D_lin_d4']=pD
print("D lin d4 MAE:", np.abs(pD-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
for c1 in ['A_lin_d6','B_log_d6','C_sqrt_d6','D_lin_d4']:
    for c2 in ['A_lin_d6','B_log_d6','C_sqrt_d6','D_lin_d4']:
        if c1<c2:
            p=.5*res[c1]+.5*res[c2]
            print("blend",c1,c2, np.abs(p-yva).mean().round(3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
allF = agent_api.load_saved("allF.parquet")
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr_days = [123,151,179,207,235,263,291,319,347,375]
tr = allF[allF.snapshot_day.isin(tr_days)]
va = allF[allF.snapshot_day==403]
Xtr, ytr = tr[feats].values, tr.future_spend_4w.values
Xva, yva = va[feats].values, va.future_spend_4w.values
w = 0.5 ** ((375 - tr.snapshot_day.values)/140.0)

def fit_q(yt, seed, depth=6, lr=0.03, n=1200):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=lr, n_estimators=n, max_depth=depth, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)

# E: linear d5
t0=time.time()
pE = np.mean([fit_q(ytr, s, depth=5) for s in (1,2,3)], axis=0)
print("E lin d5 MAE:", np.abs(pE-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# F: linear d6 longer rounds
t0=time.time()
pF = np.mean([fit_q(ytr, s, depth=6, n=2400) for s in (1,2,3)], axis=0)
print("F lin d6 n2400 MAE:", np.abs(pF-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# G: linear d6, min_child_weight 30
t0=time.time()
def fit_q2(yt, seed, mcw=30):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=0.03, n_estimators=1200, max_depth=6, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)
pG = np.mean([fit_q2(ytr, s) for s in (1,2,3)], axis=0)
print("G lin d6 mcw30 MAE:", np.abs(pG-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# blends with d4
print("blend d4+d5:", np.abs(.5*pD+.5*pE-yva).mean().round(3))
print("blend d4+d6:", np.abs(.5*pD+.5*pA-yva).mean().round(3))
print("blend d5+d6:", np.abs(.5*pE+.5*pA-yva).mean().round(3))
print("blend d4+2d6:", np.abs((pD+2*pA)/3-yva).mean().round(3))