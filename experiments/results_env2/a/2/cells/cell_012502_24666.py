import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)
df["err_sq"] = df.oof_sq - df.future_spend_4w
df["err_med"] = df.oof_med - df.future_spend_4w
df["err_log"] = df.oof_log - df.future_spend_4w
df["eq"] = (df.oof_sq+df.oof_med+df.oof_log)/3
print("\nMAE by learner (OOF):")
for c in ["oof_sq","oof_med","oof_log","eq"]:
    print(c, round(np.abs(df[c]-df.future_spend_4w).mean(),3))
print("\nMean target & mean pred & mean err by snapshot day:")
g = df.groupby("snapshot_day").agg(t=("future_spend_4w","mean"), sq=("oof_sq","mean"), med=("oof_med","mean"), log=("oof_log","mean"))
g["err_eq"] = (g.sq+g.med+g.log)/3 - g.t
print(g.round(2))
# grid search blend weights on OOF for MAE
best=None
for a in np.arange(0,1.01,0.05):
    for b in np.arange(0,1.01-a+1e-9,0.05):
        c = 1-a-b
        p = a*df.oof_sq+b*df.oof_med+c*df.oof_log
        m = np.abs(p-df.future_spend_4w).mean()
        if best is None or m<best[0]: best=(m,a,b,c)
print("\nbest OOF blend weights (sq,med,log):", best)
# equal-weight OOF MAE for reference
print("equal:", np.abs(df.eq-df.future_spend_4w).mean())
