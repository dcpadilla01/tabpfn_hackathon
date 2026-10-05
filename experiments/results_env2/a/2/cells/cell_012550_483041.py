import agent_api, pandas as pd, numpy as np
f3 = agent_api.load_saved("feats_v3.parquet")
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"]).merge(
    f3[["household_key","snapshot_day","spend_28","spend_56","spend_84","days_since_last","active_28","avg_basket_84","has_demo","week_of_year"]],
    on=["household_key","snapshot_day"])
y = df.future_spend_4w
print("OOF MAE: med", round(np.abs(df.oof_med-y).mean(),3))
# persistence baselines
for c in ["spend_28","spend_56","spend_84"]:
    print("persist", c, round(np.abs(df[c]-y).mean(),3))
print("persist blend 0.5*28+0.3*56+0.2*84:", round(np.abs((0.5*df.spend_28+0.3*df.spend_56+0.2*df.spend_84)-y).mean(),3))
# optimal global scale for med
for s in [0.95,1.0,1.05,1.1,1.15,1.2]:
    print("scale",s, round(np.abs(s*df.oof_med-y).mean(),3))
# error by segment
df["seg"] = pd.cut(df.spend_28, [-1,1,50,150,300,100000], labels=["0","low","mid","high","vhigh"])
print("\nMAE(oof_med) & mean err & n by spend_28 seg:")
print(df.groupby("seg", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "mae":np.abs(g.oof_med-g.future_spend_4w).mean(), "bias":(g.oof_med-g.future_spend_4w).mean()}), include_groups=False).round(2))
df["seg2"] = pd.cut(df.days_since_last, [-1,7,14,28,56,10000], labels=["<=7","8-14","15-28","29-56",">56"])
print("\nby days_since_last:")
print(df.groupby("seg2", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "mae":np.abs(g.oof_med-g.future_spend_4w).mean(), "bias":(g.oof_med-g.future_spend_4w).mean()}), include_groups=False).round(2))
