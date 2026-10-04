import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()
def prep(df, feats):
    X = df[["household_key","snapshot_day"]].copy()
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce")
        if x.dtype == object or df[f].dtype == bool:
            x = x.astype(float) if df[f].dtype==bool else x
        X[f] = x.astype(float).fillna(0.0) if x.notna().any() else 0.0
    return X

def ridge_eval(df, feats, train_days, val_days, alpha=300.0):
    d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
    X = prep(d, feats).values
    y = d["future_spend_4w"].values.astype(float)
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    mu = X[tr].mean(0); sg = X[tr].std(0)+1e-9
    Z = (X-mu)/sg
    A = Z[tr].T@Z[tr] + alpha*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    pred = Z[va]@w
    return np.abs(pred - y[va]).mean()

tables = ["e000_base","e001_history","e003_dept_mix","e004_long_hist","e005_seasonal_peer",
          "e006_seq_gaps","e007_new","e008_decomp2","e009_macro","e010_composite","e011_display",
          "e002_marketing","e012_full","e013_union","e014_base","cand_new"]
harness = {"e000_base":92.531,"e001_history":63.025,"e003_dept_mix":64.175,"e004_long_hist":63.982,
           "e005_seasonal_peer":67.852,"e006_seq_gaps":64.050,"e007_new":62.794,"e008_decomp2":61.711,
           "e009_macro":61.647,"e010_composite":63.939,"e011_display":61.680,"e002_marketing":64.676,
           "e012_full":61.471,"e013_union":61.337,"e014_base":63.242,"cand_new":None}

print(f"{'table':22s} {'proxy431':>9s} {'harness':>9s}")
out=[]
for n in tables:
    try:
        df = load_saved(n+".parquet")
        feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
        mae = ridge_eval(df, feats, list(range(95,404,28)), [431])
        out.append((n, mae, harness.get(n)))
        print(f"{n:22s} {mae:9.3f} {str(harness.get(n)):>9s}")
    except Exception as e:
        print(n, "ERR", repr(e))
# rank correlation
vals = [(h, m) for n,m,h in out if h is not None]
import numpy as np
h_ = np.array([v[1] for v in vals]); p_ = np.array([v[0] for v in vals])
print("spearman proxy vs harness:", np.corrcoef(h_.argsort().argsort(), p_.argsort().argsort())[0,1].round(3))
