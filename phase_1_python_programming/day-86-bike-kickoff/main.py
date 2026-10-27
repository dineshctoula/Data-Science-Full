"""Run the Day 86 bike-sharing kickoff (Project 2)."""

from pathlib import Path

import pandas as pd

from audit_engine import daily_shapes, gap_report, lag_correlations, leak_check, weather_effect, year_over_year
from baseline_engine import baselines, day_block_bootstrap, mae, score_table
from bike_data import TARGET, fetch, load_grid
from split_engine import SLICES, assign_slices, load_split_dates, scorable, write_split_dates
from visualizer import BikePlots

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SPLIT_FILE = HERE / "split_dates.tsv"
SAMPLE_WEEK = ("2012-08-06", "2012-08-12")


def run_pipeline():
    print("=" * 72)
    print("DAY 86: BIKE-SHARING KICKOFF (PROJECT 2)")
    print("=" * 72)

    fetch(DATA_DIR)
    grid = load_grid(DATA_DIR)
    print("\n1) data: UCI bike sharing, hourly, 2011-2012 (sha256 checked)")
    print(f"   {len(grid)} hours on the full grid; {int(grid['missing'].sum())} absent from the file")
    gaps = gap_report(grid)
    outages = gaps[gaps["outage"]]
    quiet = int((grid["missing"] & ~grid["outage"]).sum())
    print(f"   {quiet} scattered quiet hours filled with 0 (no recorded hour is ever 0)")
    print(f"   {len(outages)} outage days, never scored: " + ", ".join(d.strftime("%Y-%m-%d") for d in outages["date"]))

    print("\n2) audit")
    leak = leak_check(grid)
    print(f"   casual + registered == cnt on all {leak['rows']} usable hours: {leak['exact']}  -> both are dropped")
    yoy = year_over_year(grid)
    print(f"   2012 vs 2011, same month: {yoy.min():.2f}x to {yoy.max():.2f}x  -> the split must follow time")
    shapes = daily_shapes(grid)
    print(
        f"   working days peak at 8 a.m. ({shapes.loc[8, 'working day']:.0f}) and 5 p.m. ({shapes.loc[17, 'working day']:.0f}); "
        f"weekends peak mid-afternoon ({shapes['weekend / holiday'].max():.0f})"
    )
    for _, r in lag_correlations(grid).iterrows():
        note = "" if r["usable_day_ahead"] else "   (not available a day ahead)"
        print(f"   corr with the count {int(r['lag_hours']):>3} h earlier: {r['corr']:.3f}{note}")
    w = weather_effect(grid)
    print("   per-hour rentals by weather 1-4: " + ", ".join(f"{v:.0f}" for v in w["per_hour"]))

    # the cut dates are written once; after that they are only read
    if not SPLIT_FILE.exists():
        write_split_dates(SPLIT_FILE)
    ranges = load_split_dates(SPLIT_FILE)
    slices = assign_slices(grid, ranges)
    ok = scorable(grid)
    print("\n3) frozen time split (split_dates.tsv)")
    for name, r in zip(SLICES, ranges.itertuples()):
        m = (slices == name) & ok
        print(f"   {name:<10} {r.first_day:%Y-%m-%d} .. {r.last_day:%Y-%m-%d}  {int(m.sum()):5d} scorable hours, mean {grid.loc[m, TARGET].mean():5.1f}/h")

    preds = baselines(grid, slices == "train")
    val = (slices == "validation") & ok
    for p in preds.values():
        val &= p.notna()
    y = grid.loc[val, TARGET]
    table = score_table(y, {k: v[val] for k, v in preds.items()})
    print(f"\n4) day-ahead baselines on validation ({int(val.sum())} hours, {grid.loc[val, 'date'].nunique()} days; test stays closed)")
    print(f"   {'baseline':<34} {'MAE':>6}  {'RMSE':>6}  {'WAPE':>6}")
    for _, r in table.iterrows():
        print(f"   {r['baseline']:<34} {r['mae']:6.1f}  {r['rmse']:6.1f}  {r['wape']:6.1%}")
    best = table.loc[table["mae"].idxmin(), "baseline"]
    boot = day_block_bootstrap(y, preds[best][val], preds["last week, same hour"][val], grid.loc[val, "date"])
    print(f"   {best} minus last week: MAE {boot['diff']:+.1f}, 95% [{boot['lo']:+.1f}, {boot['hi']:+.1f}] (resampling {boot['days']} days)")

    week = val & (grid["date"] >= SAMPLE_WEEK[0]) & (grid["date"] <= SAMPLE_WEEK[1])
    plots = BikePlots(HERE / "output")
    paths = [
        plots.timeline(grid, ranges),
        plots.daily_shapes(shapes),
        plots.sample_week(
            grid.loc[week, "ts"],
            grid.loc[week, TARGET],
            {k: preds[k][week] for k in ("last week, same hour", "mean of last 4 weeks, same hour", "training-period profile")},
            title=f"validation week {SAMPLE_WEEK[0]} to {SAMPLE_WEEK[1]}: actual vs baselines",
        ),
    ]
    print("\n5) plots saved")
    for p in paths:
        print("  ", Path(p).relative_to(HERE))
    print(f"\nDay 86 done. The bar for the first model: validation MAE below {mae(y, preds[best][val]):.1f}.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    run_pipeline()
