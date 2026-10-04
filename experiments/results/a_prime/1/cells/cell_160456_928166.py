import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
base_cols = set(A.load_saved("e011_demo.parquet").columns)

groups = {
 "dorm": [c for c in A.load_saved("e012_dorm.parquet").columns if c not in base_cols],
 "peers": [c for c in A.load_saved("e013_peers.parquet").columns if c not in base_cols],
 "micro": [c for c in A.load_saved("micro.parquet").columns if c not in base_cols],
 "robust": [c for c in A.load_saved("nf_robust.parquet").columns if c not in base_cols],
 "seasonal": [c for c in A.load_saved("nf_seasonal.parquet").columns if c not in base_cols],
}
print({k: len(v) for k,v in groups.items()})

tt2 = tt.merge(A.load_saved("e011_demo.parquet")[["household_key","snapshot_day","spend_rate28","spend_l1","zero_recent"]],
               on=["household_key","snapshot_day"])
y = tt2["future_spend_4w"]

for g, cols in groups.items():
    df = A.load_saved({  "dorm":"e012_dorm.parquet","peers":"e013_peers.parquet","micro":"micro.parquet",
                         "robust":"nf_robust.parquet","seasonal":"nf_seasonal.parquet"}[g])
    m = tt2.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="left")
    rows=[]
    for c in cols:
        x = m[c].astype(float)
        r1 = x.corr(y)
        # partial: corr of residual after regressing on spend_rate28
        b = np.polyfit(m["spend_rate28"].fillna(0), y, 1)
        res = y - np.polyval(b, m["spend_rate28"].fillna(0))
        r2 = x.corr(res)
        rows.append((c, round(r1,3), round(r2,3)))
    rows.sort(key=lambda t: -abs(t[2]))
    print("==", g)
    for r in rows[:12]: print("  ", r)
