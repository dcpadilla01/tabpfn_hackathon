import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr = F[F.future_spend_4w.notna()].copy()
TRN = tr[tr.snapshot_day <= 375]
HOLD = tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_q(df_tr, alpha=0.5, rounds=1200, lr=0.03, half=140.0, seed=1, depth=6):
    w = 0.5 ** ((df_tr.snapshot_day.values - df_tr.snapshot_day.max()) / half)
    p = dict(objective="reg:quantileerror", quantile_alpha=alpha, eta=lr, max_depth=depth,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(df_tr[feats], label=df_tr.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return np.clip(b.predict(xgb.DMatrix(HOLD[feats])), 0, None)

t0=time.time()
pq = fit_q(TRN)
print("q alpha.5 MAE %.3f" % np.abs(pq-yv).mean())

# 1) alpha sweep (global calibration proxy)
for a in [0.45, 0.55, 0.60]:
    p = fit_q(TRN, alpha=a)
    print("alpha %.2f MAE %.3f mean %.1f" % (a, np.abs(p-yv).mean(), p.mean()))

# 2) residual boosting: fit L2 on residual of q model, using train preds
w = 0.5 ** ((TRN.snapshot_day.values - TRN.snapshot_day.max()) / 140)
p = dict(objective="reg:squarederror", eta=0.03, max_depth=5, min_child_weight=10,
         subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=2)
bq = xgb.train(dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=1),
               xgb.DMatrix(TRN[feats], label=TRN.future_spend_4w.values, weight=w), num_boost_round=1200)
ptr = bq.predict(xgb.DMatrix(TRN[feats]))
resid = TRN.future_spend_4w.values - ptr
for nr in [300, 600]:
    br = xgb.train(p, xgb.DMatrix(TRN[feats], label=resid, weight=w), num_boost_round=nr)
    pr = bq.predict(xgb.DMatrix(HOLD[feats])) + br.predict(xgb.DMatrix(HOLD[feats]))
    pr = np.clip(pr, 0, None)
    print("resid boost %d MAE %.3f mean %.1f" % (nr, np.abs(pr-yv).mean(), pr.mean()))

# 3) level-binned recalibration (fit on holdout itself = upper bound of gain)
bins = np.quantile(pq, np.linspace(0,1,11))
idx = np.clip(np.digitize(pq, bins[1:-1]), 0, 9)
corr = np.array([np.median(yv[idx==i] - pq[idx==i]) if (idx==i).sum()>5 else 0 for i in range(10)])
pc = pq + corr[idx]
print("binned recal (in-sample) MAE %.3f  (gain upper bound)" % np.abs(pc-yv).mean())
print("bin medians resid:", np.round(corr,1))
# 4) isotonic y~p
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip").fit(pq, yv)
pi = iso.predict(pq)
print("isotonic (in-sample) MAE %.3f" % np.abs(pi-yv).mean())
print("%.0fs" % (time.time()-t0))
