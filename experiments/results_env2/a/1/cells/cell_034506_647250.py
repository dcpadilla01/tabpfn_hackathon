oof = load_saved("oof_e5.parquet")
ap = load_saved("e016_allpreds.parquet")
held = load_saved("e016_held.parquet")

# target mean by snapshot day (train)
print(oof.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]).round(1))

m = oof.merge(ap[["household_key","snapshot_day","pq","pl","pa"]], on=["household_key","snapshot_day"])
print("\nOOF bias (pred-target) by day:")
print(m.groupby("snapshot_day").apply(lambda g: pd.Series({
    "tgt": g.future_spend_4w.mean(), "pq": g.pq.mean(), "pl": g.pl.mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean(), "pl_bias": (g.pl-g.future_spend_4w).mean()}), include_groups=False).round(1))

mh = held.merge(ap[["household_key","snapshot_day","pq","pl","pa"]], on=["household_key","snapshot_day"])
print("\nHeld bias by day:")
print(mh.groupby("snapshot_day").apply(lambda g: pd.Series({
    "tgt": g.future_spend_4w.mean(), "pq": g.pq.mean(), "pl": g.pl.mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean()}), include_groups=False).round(1))

# MAE by target bucket on held for pq
mh["bucket"] = pd.cut(mh.future_spend_4w, [-1,1,50,100,200,400,10000])
print("\nHeld MAE by target bucket:")
print(mh.groupby("bucket", observed=True).apply(lambda g: pd.Series({
    "n": len(g), "pq": np.abs(g.pq-g.future_spend_4w).mean(),
    "pq_bias": (g.pq-g.future_spend_4w).mean()}), include_groups=False).round(1))
