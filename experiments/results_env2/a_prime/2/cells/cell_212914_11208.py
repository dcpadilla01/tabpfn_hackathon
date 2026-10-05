h = history(1)
h = h.sort_values("day")
print(h[["day","basket_id","sales_value"]].head(15).to_string())
print("total spend:", h.sales_value.sum())

for s in [151, 179]:
    for lo, hi, name in [(s-27, s, "spend_28 [s-27,s]"), (s-28, s-1, "spend_28 [s-28,s-1]"),
                         (s-55, s, "spend_56 [s-55,s]"), (s-56, s-1, "spend_56 [s-56,s-1]"),
                         (s-83, s, "spend_84 [s-83,s]"), (s-84, s-1, "spend_84 [s-84,s-1]"),
                         (s-55, s-28, "spend_28_prior [s-55,s-28]"), (s-56, s-29, "spend_28_prior [s-56,s-29]"),
                         (s-167, s-84, "spend_84_prior [s-167,s-84]"), (s-168, s-85, "spend_84_prior [s-168,s-85]")]:
        v = h[(h.day >= lo) & (h.day <= hi)].sales_value.sum()
        print(f"s={s} {name}: {v:.2f}")
