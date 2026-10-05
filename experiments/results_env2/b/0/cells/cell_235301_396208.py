df = agent_api.load_saved("e013_stationary.parquet")
tgt = agent_api.train_targets()
m = df.merge(tgt, on=["household_key","snapshot_day"], how="left")
print(m.future_spend_4w.notna().sum(), len(tgt))

# households present in display_mailer at all (as of 459)
v = agent_api.snapshot()
dm = v.display_mailer
tx = v.transactions
hh_dm = set(dm.product_id.unique())
# link: product -> household via transactions
sub = tx[tx.product_id.isin(hh_dm)]
print("tx rows with display products:", len(sub), "hh:", sub.household_key.nunique(), "of", tx.household_key.nunique())

# does household-level display exposure correlate with future spend?
dmw = dm.copy()
txw = tx.merge(dmw, on=["product_id","store_id","week_no"], how="inner")
print("matched tx rows:", len(txw))
g = txw.groupby("household_key").agg(exp_wk=("week_no","nunique"), disp_rows=("display", lambda s:(s>0).sum()))
print(g.describe())
mm = m[m.snapshot_day==459][["household_key","future_spend_4w"]].merge(g, left_on="household_key", right_index=True, how="left")
print(mm[["future_spend_4w","exp_wk","disp_rows"]].corr().round(3))
