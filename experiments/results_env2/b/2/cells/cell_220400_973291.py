
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.abs(np.diag(G)).mean()*np.eye(n), b) if False else np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:36s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# build untapped features from transactions history
def make_feats(view):
    t = view.table("transactions")
    g = t.groupby("household_key")
    f = pd.DataFrame(index=g.size().index)
    for w in (28, 56, 84):
        tw = t[t.day > view.day - w]
        gw = tw.groupby("household_key")
        f[f"units_{w}"] = gw["quantity"].sum()
        f[f"nprod_{w}"] = gw["product_id"].nunique()
        f[f"nbask_{w}"] = gw["basket_id"].nunique()
        f[f"nstore_{w}"] = gw["store_id"].nunique()
    f["max_basket_84"] = t[t.day > view.day-84].groupby("household_key")["sales_value"].max()
    f["med_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").median()
    f["q75_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").quantile(0.75)
    f["std_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").std()
    f["spend_per_unit_84"] = f["units_84"].rdiv if False else t[t.day > view.day-84].groupby("household_key")["sales_value"].sum() / f["units_84"].replace(0,np.nan)
    return f

from agent_api import build_features
F = build_features(make_feats)
M = m.merge(F, on=["household_key","snapshot_day"], how="left")
new_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
print("new feats:", len(new_cols), "| NaN frac:", M[new_cols].isna().mean().round(2).to_dict())
run(pd.concat([B, M[new_cols].astype(float)], axis=1), "+ units/variety/basket-dist")
run(M[new_cols].astype(float), "units/variety/basket-dist alone")
