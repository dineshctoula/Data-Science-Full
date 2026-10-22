"""Run the Day 81 churn-project kickoff."""

from pathlib import Path

import pandas as pd

from audit_engine import charges_check, churn_by, column_audit, shadow_levels, tenure_buckets
from baseline_engine import TOP_FRAC, RateTable, add_tenure_band, run_baselines
from churn_data import TARGET, load_clean, load_raw, fetch
from split_engine import load_split, stratified_split, write_split
from visualizer import KickoffPlots

HERE = Path(__file__).resolve().parent
SPLIT_FILE = HERE / "split_ids.tsv"


def run_pipeline():
    print("=" * 72)
    print("DAY 81: CHURN PROJECT KICKOFF")
    print("=" * 72)

    path = fetch(HERE / "data")
    raw = load_raw(path)
    df = load_clean(HERE / "data")
    print("\n1) data")
    print(f"   {path.name}: {len(raw)} rows, {raw.shape[1]} columns, sha256 checked")
    blank_total = int((raw["TotalCharges"].str.strip() == "").sum())
    print(f"   TotalCharges blank for {blank_total} rows, all tenure 0 -> filled with 0")
    print(f"   churn rate {df[TARGET].mean():.3f} ({df[TARGET].sum()} of {len(df)})")

    audit = column_audit(df)
    print("\n2) audit")
    print(f"   missing after cleaning: {int(audit['missing'].sum())}")
    lopsided = audit[(audit["kind"] == "category") & (audit["top_share"] > 0.8)]
    for _, row in lopsided.iterrows():
        print(f"   {row['column']} is {row['top_share']:.0%} '{row['top']}'")
    for group in shadow_levels(df):
        cols = ", ".join(col for col, _ in group)
        print(f"   same rows, {len(group)} names: {cols}")
    charges = charges_check(df)
    print(
        f"   TotalCharges vs tenure × MonthlyCharges: corr {charges['corr']:.4f}, "
        f"median gap {charges['median_rel_gap']:.1%}"
    )

    print("\n3) churn by the columns that move it most")
    panels = []
    for col in ("Contract", "InternetService", "PaymentMethod"):
        table = churn_by(df, col)
        panels.append((col, table))
        print("   " + ",  ".join(f"{lvl} {r:.1%}" for lvl, r in zip(table[col], table["rate"])))
    bands = tenure_buckets(df)
    panels.append(("tenure (months)", bands))
    print("   tenure " + ",  ".join(f"{b} {r:.1%}" for b, r in zip(bands["band"], bands["rate"])))

    # the split file is written once; after that it is only ever read
    if not SPLIT_FILE.exists():
        write_split(df, stratified_split(df, seed=81), SPLIT_FILE)
        print(f"\n   wrote {SPLIT_FILE.name}")
    parts = load_split(df, SPLIT_FILE)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    print("\n4) frozen split (split_ids.tsv)")
    for name, part in parts.items():
        print(f"   {name:<10} {len(part):5d} rows, churn {part[TARGET].mean():.3f}")

    results = run_baselines(train, val)
    n_leavers = int(val[TARGET].sum())
    ceiling = min(1.0, TOP_FRAC * len(val) / n_leavers)
    print(f"\n5) baselines on validation ({len(val)} rows, {n_leavers} leavers; test stays closed)")
    print(f"   calling the top {TOP_FRAC:.0%} means {TOP_FRAC * len(val):.0f} calls; a perfect list reaches {ceiling:.1%}")
    print(f"   {'baseline':<44} {'cells':>5}  {'top20':>6}  {'AUC':>6}  {'log loss':>8}")
    for _, r in results.iterrows():
        print(f"   {r['baseline']:<44} {r['cells']:5d}  {r['top20_recall']:6.3f}  {r['auc']:6.3f}  {r['log_loss']:8.4f}")

    band_train, band_val = add_tenure_band(train), add_tenure_band(val)
    scores = {
        "everyone at the train rate": RateTable([]).fit(band_train).predict(band_val),
        "contract": RateTable(["Contract"]).fit(band_train).predict(band_val),
        "contract × tenure × internet": RateTable(["Contract", "tenure_band", "InternetService"]).fit(band_train).predict(band_val),
        # the best a list could ever do: every leaver first
        "perfect list": val[TARGET].to_numpy(dtype=float),
    }

    plots = KickoffPlots(HERE / "output")
    p1 = plots.churn_profile(panels, overall=float(df[TARGET].mean()))
    p2 = plots.gains(val[TARGET], scores, frac=TOP_FRAC)
    print("\n6) plots saved")
    for p in (p1, p2):
        print("  ", Path(p).relative_to(HERE))
    bar = results["top20_recall"].max()
    print(f"\nDay 81 done. The bar for the first model: beat {bar:.3f} top-20% recall on validation.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    run_pipeline()
