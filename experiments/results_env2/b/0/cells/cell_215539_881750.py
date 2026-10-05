import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])

# pseudo-val: fit ridge on snapshots <=403, eval on 431
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
X = pd.get_dummies(m[feat].fillna(-1), columns=["class2"], dummy_na=True)
X = X.astype(float)
mu, sd = X.mean(), X.std().replace(0,1)
Xz = (X-mu)/sd
tr_mask = m.snapshot_day <= 403
va_mask = m.snapshot_day == 431
Xtr, ytr = Xz[tr_mask].values, m.future_spend_4w[tr_mask].values
Xva, yva = Xz[va_mask].values, m.future_spend_4w[va_mask].values
I = np.eye(Xz.shape[1])
for lam in [1e2, 1e3, 1e4]:
    w = np.linalg.solve(Xtr.T@Xtr + lam*I, Xtr.T@ytr)
    p = Xva@w
    mae = float(np.abs(yva-p).mean())
    r2 = float(1 - ((yva-p)**2).sum()/((yva-yva.mean())**2).sum())
    print(f"ridge lam={lam}: pseudo-val MAE {mae:.2f} R2 {r2:.3f}")

# simple heuristics on 431
val = m[va_mask]
y = val.future_spend_4w.values
s28 = val["spend_28"].fillna(0).values
s56 = val["spend_56"].fillna(0).values
s84 = val["spend_84"].fillna(0).values
print("MAE spend28:", round(float(np.abs(y-s28).mean()),2))
for k in [0.8,0.9,1.0]:
    print(f"MAE {k}*spend28:", round(float(np.abs(y-k*s28).mean()),2))
best = min(((round(float(a),2), round(float(np.abs(y-(a*s28+(1-a)*(s56-s28))).mean()),2)) for a in np.linspace(0,1,11)), key=lambda z:z[1])
print("best blend spend28 vs (spend56-spend28):", best)
# residual analysis of ridge best model
lam=1e3
w = np.linalg.solve(Xtr.T@Xtr + lam*I, Xtr.T@ytr)
p = Xva@w
res = yva - p
print("\nresid mean:", round(float(res.mean()),2), "MAE:", round(float(np.abs(res).mean()),2))
# error by target bucket
for lo,hi in [(0,1),(1,50),(50,150),(150,300),(300,10**9)]:
    msk = (yva>=lo)&(yva<hi)
    if msk.sum()>0:
        print(f"y in [{lo},{hi}): n={msk.sum()}, MAE={float(np.abs(res[msk]).mean()):.1f}, mean_pred={float(p[msk].mean()):.1f}, mean_y={float(yva[msk].mean()):.1f}")
