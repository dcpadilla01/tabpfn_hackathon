df = agent_api.load_saved("e013_stationary.parquet")
tgt = agent_api.train_targets()
v = agent_api.snapshot()
dm = v.display_mailer.copy()
tx = v.transactions.copy()
dm["display"] = dm.display.astype(int)
txw = tx.merge(dm, on=["product_id","store_id","week_no"], how="inner")
print("matched tx rows:", len(txw), "hh:", txw.household_key.nunique())

g = txw.groupby("household_key").agg(
    exp_wk=("week_no","nunique"),
    disp_rows=("display", lambda s: int((s.astype(int) > 0).sum())),
)
m = df.merge(tgt, on=["household_key","snapshot_day"], how="left")
mm = m[m.snapshot_day==459][["household_key","future_spend_4w"]].merge(g, left_on="household_key", right_index=True, how="left")
print(mm[["future_spend_4w","exp_wk","disp_rows"]].corr().round(3))
print(mm.head())
