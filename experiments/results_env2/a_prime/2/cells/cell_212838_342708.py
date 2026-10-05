e1 = load_saved("e001_recent_behavior.parquet")
print("e1 cols:", e1.columns.tolist())
print("e1 shape:", e1.shape)
print(e1.head(2).T)

tt = train_targets()
print("\ntargets shape:", tt.shape)
print(tt.future_spend_4w.describe())
m = tt.merge(e1, on=["household_key","snapshot_day"], how="left")
num = m.select_dtypes(include=[np.number]).columns
print("\ncorr with target:")
print(m[num].corr()["future_spend_4w"].sort_values())

def probe(view, s):
    try:
        t = load_saved("e001_recent_behavior.parquet")
        ok = "OK " + str(t.shape)
    except NameError:
        try:
            t = agent_api.load_saved("e001_recent_behavior.parquet")
            ok = "OK-via-agent_api " + str(t.shape)
        except Exception as ex:
            ok = f"FAIL {type(ex).__name__}: {ex}"
    except Exception as ex:
        ok = f"FAIL {type(ex).__name__}: {ex}"
    print("snapshot", s, "| load_saved inside fn:", ok, "| n_hh:", len(view.households))
    return pd.DataFrame(index=pd.Index(view.households, name="household_key"))

bf = build_features(probe)
print("build_features out:", bf.shape)
