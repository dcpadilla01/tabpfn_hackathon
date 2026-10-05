
import pandas as pd, numpy as np
from agent_api import load_saved

names = ["e001_preds","e002_preds","e003_preds","e004_preds","e005_preds","e007_preds",
         "e008_preds","e009_preds","e010_preds","e011_preds","e012_preds","e014_preds",
         "e016_preds","e017_preds","e018_preds"]
tabs = {}
for n in names:
    df = load_saved(n + ".parquet")
    tabs[n] = df
    print(n, df.shape, list(df.columns))

base = tabs["e005_preds"][["household_key","snapshot_day"]]
for n, df in tabs.items():
    m = df.merge(base, on=["household_key","snapshot_day"], how="inner")
    print(n, "aligned:", len(m), "/", len(base))

P = pd.DataFrame({n: df.set_index(["household_key","snapshot_day"])["prediction"]
                  for n, df in tabs.items()})
print("\ncorr matrix:")
print(P.corr().round(3))
print("\nsummary:")
print(P.describe().T[["mean","std","min","max"]].round(2))

oof = load_saved("oof_e5.parquet")
print("\noof_e5:", oof.shape, list(oof.columns))
print(oof.head(3))
rep = load_saved("repro_e5.parquet")
print("repro_e5:", rep.shape, list(rep.columns))
print(rep.head(3))


# ---- cell ----

import pandas as pd, numpy as np
from agent_api import load_saved

oof = load_saved("oof_e5.parquet")
oof["resid"] = oof["future_spend_4w"] - oof["pred"]
oof["abs_err"] = oof["resid"].abs()
print("OOF rows:", len(oof), " OOF MAE: %.3f" % oof["abs_err"].mean(),
      " mean pred %.2f mean target %.2f" % (oof["pred"].mean(), oof["future_spend_4w"].mean()))
g = oof.groupby("snapshot_day").agg(n=("resid","size"), mae=("abs_err","mean"),
        bias=("resid","mean"), mp=("pred","mean"), mt=("future_spend_4w","mean"))
print(g.round(2))

# bias trend vs snapshot day (linear fit)
X = np.vstack([np.ones(len(oof)), oof["snapshot_day"]]).T
y = oof["resid"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print("resid ~ const %.3f + %.4f*day" % (beta[0], beta[1]))
# weighted by decay toward validation days (459..543): weight ~ 0.5**((459-day)/140)
w = 0.5 ** (np.maximum(0, 459 - oof["snapshot_day"]) / 140.0)
Xw = X * w[:,None]
bw, *_ = np.linalg.lstsq(Xw, y*w, rcond=None)
print("weighted: const %.3f + %.4f*day -> correction at day 459: %.2f, 543: %.2f" %
      (bw[0], bw[1], bw[0]+bw[1]*459, bw[0]+bw[1]*543))

allF = load_saved("allF.parquet")
print("\nallF:", allF.shape)
print([c for c in allF.columns][:60])


# ---- cell ----

import pandas as pd, numpy as np
from agent_api import load_saved, save_table

# Full diagnostics (redo truncated parts)
oof = load_saved("oof_e5.parquet"); oof["ae"]=(oof.future_spend_4w-oof.pred).abs()
print("OOF MAE by snapshot (incl 403):")
print(oof.groupby("snapshot_day")["ae"].agg(["size","mean"]).round(2).tail(4))

allF = load_saved("allF.parquet")
print("\nallF:", allF.shape, "| val rows:", (allF.snapshot_day>=459).sum(), "| train rows:", (allF.snapshot_day<459).sum())
print("n features:", allF.shape[1]-2)

names = ["e004_preds","e005_preds","e011_preds","e018_preds","e016_preds","e009_preds","e003_preds"]
P = {n: load_saved(n+".parquet").set_index(["household_key","snapshot_day"])["prediction"] for n in names}
C = pd.DataFrame(P).corr()
print("\ncorr among top tables:")
print(C.round(4).to_string())

# E019: equal-weight blend of the 4 best distinct recipes
blend = (P["e005_preds"] + P["e011_preds"] + P["e004_preds"] + P["e018_preds"]) / 4.0
out = blend.reset_index(); out.columns = ["household_key","snapshot_day","prediction"]
p = save_table(out, "e019_blend_preds")
print("\nsaved:", p, out.shape)
print(out["prediction"].describe().round(2))
