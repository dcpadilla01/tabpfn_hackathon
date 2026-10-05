
import agent_api, pandas as pd, numpy as np
pred = agent_api.load_saved('pred_e011.parquet')
tt = agent_api.train_targets()
feats = agent_api.load_saved('feats_v4.parquet')
val = pred[pred.snapshot_day>=459]
mv = val.merge(tt, on=['household_key','snapshot_day'])
print("val rows:", len(mv), "MAE:", np.abs(mv.prediction-mv.future_spend_4w).mean())
mv['err'] = mv.prediction - mv.future_spend_4w
mv['abserr'] = mv.err.abs()
# error by target bucket
mv['tb'] = pd.cut(mv.future_spend_4w, [-1,0.01,25,75,150,300,600,1e9])
print(mv.groupby('tb', observed=True).agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), med_pred=('prediction','median')))
# how much MAE from top decile of predictions
q = mv.prediction.quantile(0.9)
hi = mv[mv.prediction>q]
print("top-decile-pred share of total abs err:", hi.abserr.sum()/mv.abserr.sum(), "| their MAE:", hi.abserr.mean(), "| bias:", hi.err.mean())
# zero-target rows
z = mv[mv.future_spend_4w==0]
print("zero-target: n", len(z), "MAE", z.abserr.mean(), "mean pred", z.prediction.mean(), "median pred", z.prediction.median())
# clip test
for cap in [400, 600, 800, 1000, 1200]:
    p = mv.prediction.clip(upper=cap)
    print(f"clip@{cap}: MAE {np.abs(p-mv.future_spend_4w).mean():.3f}")
# global scaling
for s in [0.9, 0.95, 1.0, 1.05]:
    print(f"scale {s}: MAE {np.abs(mv.prediction*s-mv.future_spend_4w).mean():.3f}")
# blend with exp4w_blend at various weights
mv2 = mv.merge(feats[['household_key','snapshot_day','exp4w_blend']], on=['household_key','snapshot_day'])
for w in [0.0,0.2,0.3,0.4,0.5]:
    print(f"w_model {w}: MAE {np.abs(w*mv2.prediction+(1-w)*mv2.exp4w_blend-mv2.future_spend_4w).mean():.3f}")
