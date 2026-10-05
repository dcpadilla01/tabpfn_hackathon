import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ap = load_saved("e016_allpreds.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]

tr = ap.merge(oof, on=["household_key","snapshot_day"])
he = ap.merge(held, on=["household_key","snapshot_day"])
print("tr", tr.shape, "days", sorted(tr.snapshot_day.unique()))
print("he", he.shape, "days", sorted(he.snapshot_day.unique()))
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()
print("pq train MAE (in-sample?):", round(mae(tr.pq, tr.future_spend_4w),3))
print("pq held  MAE:", round(mae(he.pq, he.future_spend_4w),3))

ytr, ptr, dtr = tr.future_spend_4w.values, tr.pq.values, tr.snapshot_day.values
yhe, phe, dhe = he.future_spend_4w.values, he.pq.values, he.snapshot_day.values

# C1 constant shift (MAE-optimal = median residual)
c1 = np.median(ytr - ptr)
# C2 multiplicative: grid
grid = np.arange(0.95, 1.35, 0.005)
maes = [mae(a*ptr, ytr) for a in grid]
a2 = grid[int(np.argmin(maes))]
# C3 isotonic target|pred
iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
# C4 isotonic + linear day-trend of residual
res = ytr - iso.predict(ptr)
daymed = tr.assign(r=res).groupby("snapshot_day").r.median()
sl, ic = np.polyfit(daymed.index.values, daymed.values, 1)
print("day-trend of residual: slope %.4f intercept %.2f" % (sl, ic))
# C5 binned median residual vs pred (non-monotone correction)
qb = np.quantile(ptr, np.linspace(0,1,21)); qb[0]-=1; qb[-1]+=1
bins = pd.cut(ptr, qb)
bmed = tr.assign(r=res).groupby(bins, observed=True).r.median()
bctr = np.array([(b.left+b.right)/2 for b in bmed.index])
# C6 isotonic weighted toward later days (half-life 140d from day 375)
w = 0.5**((375-dtr)/140)
iso6 = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr, sample_weight=w)

def apply_c5(p):
    corr = np.interp(p, bctr, bmed.values)
    return p + corr

cands = {
 "C0 identity": phe,
 "C1 const+%.1f"%c1: phe+c1,
 "C2 mult %.3f"%a2: a2*phe,
 "C3 isotonic": iso.predict(phe),
 "C4 iso+daytrend": iso.predict(phe) + (sl*dhe+ic),
 "C5 binnedresid": apply_c5(phe),
 "C6 iso_wdecay": iso6.predict(phe),
}
print("\nHELD MAE (fit on train 95-375):")
for k,v in cands.items(): print(" ", k, round(mae(v,yhe),3))

# temporal honesty check: fit on 95-291, eval on 319-375
mtr = dtr <= 291; mev = dtr >= 319
yt, pt = ytr[mtr], ptr[mtr]; ye, pe = ytr[mev], ptr[mev]
c1t = np.median(yt-pt)
isoT = IsotonicRegression(out_of_bounds="clip").fit(pt, yt)
resT = yt - isoT.predict(pt)
dmedT = tr[mtr].assign(r=resT).groupby("snapshot_day").r.median()
slT, icT = np.polyfit(dmedT.index.values, dmedT.values, 1)
print("\nTemporal check (fit 95-291 -> eval 319-375):")
print("  identity:", round(mae(pe,ye),3))
print("  const:", round(mae(pe+c1t,ye),3))
print("  isotonic:", round(mae(isoT.predict(pe),ye),3))
print("  iso+daytrend:", round(mae(isoT.predict(pe)+slT*319+icT, ye),3), "(day set to 319)")
