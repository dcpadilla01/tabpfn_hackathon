held = load_saved("e016_held.parquet")
names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds"]
print("Held (403,431) MAE by model:")
for n in names:
    m = held.merge(P[n], on=["household_key","snapshot_day"])
    print(n, round(np.abs(m.prediction-m.future_spend_4w).mean(),3))

ap = load_saved("e016_allpreds.parquet")
mh = held.merge(ap, on=["household_key","snapshot_day"])
for c in ["pq","pl","pc","pa"]:
    print("held", c, round(np.abs(mh[c]-mh.future_spend_4w).mean(),3))
# blends on held
def bl(*args):
    cs=[c for c in args]
    p = sum(mh[c] for c in cs)/len(cs)
    return np.abs(p-mh.future_spend_4w).mean()
print("blend pq+pl held:", round(bl("pq","pl"),3))
print("blend pq+pl+pa held:", round(bl("pq","pl","pa"),3))
