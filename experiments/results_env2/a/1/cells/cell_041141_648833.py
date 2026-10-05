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
