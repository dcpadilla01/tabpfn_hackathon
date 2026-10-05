
def probe(view, T):
    print("inside fn: day", view.day, "households type:", type(view.households))
    hs = view.households
    if hs is not None:
        print("households sample:", hs[:5] if hasattr(hs, "__getitem__") else hs)
    return pd.DataFrame({"x": 1.0}, index=pd.Index(sorted(view.transactions.household_key.unique()), name="household_key"))

out = build_features(probe)
print(out.shape, out.columns.tolist())
