def tiny(view, snapshot_day):
    hh = pd.Index(view.households)
    out = pd.DataFrame({"tiny": np.ones(len(hh))}, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(tiny)
print("tiny ok", f.shape)
