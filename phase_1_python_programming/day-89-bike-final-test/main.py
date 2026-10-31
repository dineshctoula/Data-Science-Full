"""Run Project 2's final test — or rehearse it on validation first.

    python3 main.py --rehearse   fit on train, score validation (test untouched)
    python3 main.py              fit on train + validation, open test once
"""

import argparse

import numpy as np

from project_paths import CHURN_FINAL, DATA_DIR, HERE, KICKOFF, SPLIT_FILE, load_from
import final_recipe as R
from baseline_engine import mae
from bike_data import TARGET, load_grid
from final_engine import card, primary_gaps, run_models
from split_engine import assign_slices, load_split_dates
from visualizer import FinalPlots

gate = load_from(CHURN_FINAL, "holdout_gate")
BikePlots = load_from(KICKOFF, "visualizer").BikePlots

LOCK = HERE / "test_opened.lock"
# the lock covers the recipe and the code that turns it into numbers
RECIPE_FILES = (HERE / "final_recipe.py", HERE / "final_engine.py")
REHEARSAL_WEEK = ("2012-08-06", "2012-08-12")


def _print_card(name, frame):
    print(f"\n   {name}")
    print(f"   {'model':<50} {'MAE':>6}  {'95% (days)':>15}  {'RMSE':>6}  {'WAPE':>6}")
    for _, r in frame.iterrows():
        spread = f" ±{r['seed_sd']:.1f} over draws" if r["seed_sd"] else ""
        print(f"   {r['model']:<50} {r['mae']:6.1f}  [{r['lo']:5.1f}, {r['hi']:5.1f}]  {r['rmse']:6.1f}  {r['wape']:6.1%}{spread}")


def _print_gaps(gaps, against):
    better = int((gaps["hi"] < 0).sum())
    worse = int((gaps["lo"] > 0).sum())
    print(
        f"   primary − {against}: mean {gaps['diff'].mean():+.1f}, range [{gaps['diff'].min():+.1f}, {gaps['diff'].max():+.1f}] "
        f"over {len(gaps)} draws; interval wholly below 0 in {better}, wholly above 0 in {worse}"
    )


def run_pipeline(rehearse: bool = False):
    print("=" * 72)
    print("DAY 89: PROJECT 2 FINAL TEST" + ("  (REHEARSAL ON VALIDATION)" if rehearse else ""))
    print("=" * 72)

    grid = load_grid(DATA_DIR)
    slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
    recipe_fp = gate.fingerprint(RECIPE_FILES)
    print(f"\n1) recipe {recipe_fp[:12]}…: primary '{R.PRIMARY}', comparison '{R.COMPARISON}'")
    print(f"   lam {R.LAM:g}; {len(R.SEEDS)} simulated forecast seasons ({R.FORECAST.name}); drift fallback: {R.DRIFT_FALLBACK}")

    # the validation card is always the train-only fit, as Days 87-88 reported it
    val_run = run_models(grid, slices == "train", slices == "validation", R)
    val_card = card(grid, val_run, R)
    cards = {"validation (train fit)": val_card}

    if rehearse:
        run, held_name, week = val_run, "validation", REHEARSAL_WEEK
        out = HERE / "output" / "rehearsal"
    else:
        held, first_time = gate.open_test({"test": slices == "test"}, LOCK, recipe_fp)
        print(f"   test opened {'for the first time; lock written' if first_time else 'again with the identical recipe'}")
        # every choice is made, so validation joins the fit hours
        run = run_models(grid, slices.isin(R.FIT_ON), held, R)
        cards["test (final fit)"] = card(grid, run, R)
        held_name, week = "test", R.SAMPLE_WEEK
        out = HERE / "output"

    n_days = grid.loc[run.scored, "date"].nunique()
    print(f"\n2) scored {held_name} hours: {int(run.scored.sum())} on {n_days} days (outage days excluded)")
    for name, frame in cards.items():
        _print_card(name, frame)

    print(f"\n3) paired gaps on {held_name} (day-block bootstrap, one interval per draw)")
    gaps = primary_gaps(grid, run, R, R.COMPARISON)
    _print_gaps(gaps, R.COMPARISON)
    _print_gaps(primary_gaps(grid, run, R, "4-week baseline"), "4-week baseline")

    y = grid.loc[run.scored, TARGET]
    month = grid.loc[run.scored, "ts"].dt.strftime("%Y-%m")
    # per-draw MAE, then averaged: averaging the draws' *predictions* would be an
    # ensemble of ten forecasts nobody has at midnight, and would flatter the primary
    print(f"\n4) {held_name} by month (primary = mean of each draw's own MAE)")
    for m in sorted(month.unique()):
        rows = (month == m).to_numpy()
        primary = np.mean([mae(y[rows], p[rows]) for p in run.primary_by_seed.values()])
        print(
            f"   {m}: baseline {mae(y[rows], run.preds['4-week baseline'][rows]):5.1f}   no weather {mae(y[rows], run.preds['no weather'][rows]):5.1f}   "
            f"primary {primary:5.1f}   actual weather {mae(y[rows], run.preds['weather, actual (upper bound)'][rows]):5.1f}"
        )
    total = {k: float(v.sum() / y.sum()) for k, v in run.preds.items()}
    total["primary (mean over draws)"] = float(np.mean([p.sum() / y.sum() for p in run.primary_by_seed.values()]))
    print("   total predicted / actual rentals: " + ", ".join(f"{k} {v:.3f}" for k, v in total.items()))

    in_week = run.scored & (grid["date"] >= week[0]) & (grid["date"] <= week[1])
    seed0 = run.primary_by_seed[R.SEEDS[0]]
    paths = [
        FinalPlots(out).cards(cards),
        FinalPlots(out).seed_gaps(gaps, R.COMPARISON),
        BikePlots(out).sample_week(
            grid.loc[in_week, "ts"],
            grid.loc[in_week, TARGET],
            {
                "4-week baseline": run.preds["4-week baseline"].reindex(grid.index[in_week]),
                "no weather": run.preds["no weather"].reindex(grid.index[in_week]),
                "primary (forecast draw 0)": seed0.reindex(grid.index[in_week]),
            },
            title=f"{held_name} week {week[0]} to {week[1]}",
        ),
    ]
    print("\n5) plots saved")
    for p in paths:
        print("  ", p.relative_to(HERE))
    print("\nRehearsal done; test untouched." if rehearse else "\nDay 89 done. Project 2's test slice has been used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rehearse", action="store_true", help="score validation instead of opening test")
    run_pipeline(rehearse=parser.parse_args().rehearse)
