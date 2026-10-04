import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved, train_targets

t = train_targets(); KEY = ["household_key","snapshot_day"]

def prep(df, feats, tr):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        med = np.nanmedian(x[tr]) if tr is not None else np.nanmedian(x)
        x = np.where(np.isnan(x), med if med==med else 0.0, x)
        cols.append(x)
    return np.column_stack(cols)

def ridge_eval(df, feats, train_days, val_days, alpha):
    d = df.merge(t, on=KEY, how="inner")
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    X = prep(d, feats, tr); y = d["future_spend_4w"].values.astype(float)
    mu = X[tr].mean(0); sg = X[tr].std(0)+1e-9
    Z = (X-mu)/sg
    A = Z[tr].T@Z[tr] + alpha*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    return np.abs(Z[va]@w - y[va]).mean()

harness = {"e009_macro":61.647,"e013_union":61.337,"e014_base":63.242,"e008_decomp2":61.711,
           "e012_full":61.471,"e001_history":63.025,"e011_display":61.680,"e010_composite":63.939}
tabs = ["e013_union","e014_base","e009_macro","e012_full","e001_history","e008_decomp2","e011_display","e010_composite"]
for alpha in [30, 300, 3000, 10000]:
    line = []
    for n in tabs:
        df = load_saved(n+".parquet")
        feats = [c for c in df.columns if c not in KEY]
        mae = ridge_eval(df, feats, list(range(95,404,28)), [403,431], alpha)
        line.append(f"{n.replace('_macro','').replace('_union','').replace('_base','').replace('_full','').replace('_history','').replace('_decomp2','').replace('_display','').replace('_composite','')}:{mae:.1f}")
    print(f"alpha={alpha:6d}  " + "  ".join(line))
print("harness:        e013:61.3  e014:63.2  e009:61.6  e012:61.5  e001:63.0  e008:61.7  e011:61.7  e010:63.9")
