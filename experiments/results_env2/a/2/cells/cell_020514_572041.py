import agent_api as A, pandas as pd, numpy as np, itertools
tt = A.train_targets()
# per-snapshot target stats (drift check)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda s:(s==0).mean()])
g.columns=["mean","median","zero_share"]
print(g.round(2))

# saved val predictions: correlations & known val MAEs
preds = {}
maes = {"e004":62.332,"e005":62.014,"e006":61.963,"e008":61.770,"e009":61.828,"e010":61.890,"e012":62.177,"e013":61.698,"e014":61.915}
files = {"e004":"pred_e004","e005":"pred_e005","e006":"pred_e006","e008":"pred_e008","e009":"pred_e009",
         "e010":"pred_e010_blend","e012":"pred_log1p","e013":"pred_seasonal","e014":"pred_e014"}
for k,f in files.items():
    d = A.load_saved(f+".parquet")
    preds[k] = d.set_index(["household_key","snapshot_day"]).prediction
P = pd.DataFrame(preds)
C = P.corr(method="spearman")
print("\nSpearman corr of val predictions (min/max off-diag):")
off = C.where(~np.eye(len(C),dtype=bool))
print("min", round(off.min().min(),4), "max", round(off.max().max(),4))
print(C.round(4).to_string())

# dispersion of predictions vs each other (std of preds per row)
print("\nrow-wise std of predictions: mean", round(P.std(axis=1).mean(),2))
print("\nPer-model val MAE (known):", maes)
# best simple pairwise blends by weighted-avg-of-known-MAE lower bound proxy is weak;
# instead check agreement: rows where models disagree most
print("\npred describe e008:", P.e008.describe().round(1).to_dict())
