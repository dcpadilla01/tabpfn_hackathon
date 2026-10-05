E2 = load_saved("e002_full.parquet")

def fn(view, s):
    d = E2[E2.snapshot_day == s].drop(columns=["snapshot_day"]).set_index("household_key")
    d = d.reindex(pd.Index(view.households, name="household_key"))
    return d

bf = build_features(fn)
ref = load_saved("e002_full.parquet")
chk = bf.merge(ref, on=["household_key","snapshot_day"], suffixes=("_b","_r"))
cols = [c for c in ref.columns if c not in ("household_key","snapshot_day")]
diffs = {c: float(np.nanmax(np.abs(chk[c+"_b"].astype(float) - chk[c+"_r"].astype(float)))) for c in cols if np.issubdtype(ref[c].dtype, np.number)}
print("max abs diffs vs saved e002_full:", max(diffs.values()))
print("bf shape:", bf.shape)
print("NaN counts (first 5):", bf.isna().sum().sort_values(ascending=False).head(5).to_dict())
