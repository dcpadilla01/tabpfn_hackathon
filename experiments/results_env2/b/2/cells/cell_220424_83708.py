
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot, build_features

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
        w = np.linalg.solve(G + a*np.eye(n), b)
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

def make_feats(view, snapshot_day):
    t = view.table("transactions")
    day = view.day
    g = t.groupby("household_key")
    f = pd.DataFrame(index=g.size().index)
    for w in (28, 56, 84):
        tw = t[t.day > day - w]
        gw = tw.groupby("household_key")
        f[f"units_{w}"] = gw["quantity"].sum()
        f[f"nprod_{w}"] = gw["product_id"].nunique()
        f[f"nstore_{w}"] = gw["store_id"].nunique()
    b84 = t[t.day > day-84].groupby(["household_key","basket_id"])["sales_value"].sum()
    f["max_basket_84"] = b84.groupby("household_key").max()
    f["med_basket_84"] = b84.groupby("household_key").median()
    f["q75_basket_84"] = b84.groupby("household_key").quantile(0.75)
    f["std_basket_84"] = b84.groupby("household_key").std()
    u84 = t[t.day > day-84].groupby("household_key")["quantity"].sum()
    f["spend_per_unit_84"] = t[t.day > day-84].groupby("household_key")["sales_value"].sum() / u84.replace(0,np.nan)
    return f

F = build_features(make_feats)
M = m.merge(F, on=["household_key","snapshot_day"], how="left")
new_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
print("new feats:", len(new_cols))
run(pd.concat([B, M[new_cols].astype(float)], axis=1), "+ units/variety/basket-dist")
run(M[new_cols].astype(float), "units/variety/basket-dist alone")
