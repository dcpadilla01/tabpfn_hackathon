
import agent_api as api

for name in ["e011_discounts", "e010_lifecycle", "e005_decay_gapcv", "e004_temporal", "e001_recent_behavior"]:
    df = api.load_saved(name + ".parquet")
    print(name, df.shape)
    print(list(df.columns))
    print("---")

tt = api.train_targets()
y = tt["future_spend_4w"]
print("train rows:", len(tt), "zero share:", (y == 0).mean(), "mean:", y.mean(), "median:", y.median(), "p90:", y.quantile(.9))
print("calls used so far: see counter")
