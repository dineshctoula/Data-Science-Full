"""Run the Day 88 forecast-weather stress test."""

import numpy as np

from project_paths import DATA_DIR, HERE, KICKOFF, SPLIT_FILE, load_from
from baseline_engine import day_block_bootstrap
from bike_data import TARGET, load_grid
from split_engine import assign_slices, load_split_dates
from stress_engine import LAM, MODES, reference_runs, run_scenario, summarize
from visualizer import StressPlots
from weather_noise import SCENARIOS, perturb, realized_errors

BikePlots = load_from(KICKOFF, "visualizer").BikePlots
SEEDS = range(10)
SAMPLE_WEEK = ("2012-08-06", "2012-08-12")


def run_pipeline():
    print("=" * 72)
    print("DAY 88: DOES THE WEATHER GAIN SURVIVE A FORECAST?")
    print("=" * 72)

    grid = load_grid(DATA_DIR)
    slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
    refs = reference_runs(grid, slices)
    print(f"\n1) Day 87 fixed points (ridge lam {LAM:g}); test stays closed")
    for name in ("no weather", "perfect"):
        print(f"   {name:<12} validation MAE {refs[name]['val_mae']:5.1f}   rolling CV MAE {refs[name]['cv_mae']:5.1f}")

    print("\n2) simulated forecast error (assumptions; one draw read back)")
    for sc in SCENARIOS:
        e = realized_errors(grid, perturb(grid, sc, seed=0))
        print(
            f"   {sc.name:<17} temp off by {e['temp_mae_c']:.1f} °C on average, humidity {e['hum_mae_pp']:.1f} pp, "
            f"{e['rain_hit']:.0%} of rain hours caught, {e['rain_false']:.1%} false alarms"
        )

    runs = []
    for sc in SCENARIOS:
        runs += run_scenario(grid, slices, sc, SEEDS)
    summary = summarize(runs, refs)
    print(f"\n3) {len(SEEDS)} noise draws per scenario; MAE is the mean over draws (± sd)")
    print(f"   {'scenario':<17} {'trained on':<11} {'validation':>15} {'kept':>6}   {'rolling CV':>15} {'kept':>6}   {'rain ×':>6}")
    for _, r in summary.iterrows():
        mode = r["mode"].replace("trained on ", "")
        print(
            f"   {r['scenario']:<17} {mode:<11} {r['val_mae']:7.1f} ± {r['val_mae_sd']:4.1f} {r['val_mae_kept']:6.0%}"
            f"   {r['cv_mae']:7.1f} ± {r['cv_mae_sd']:4.1f} {r['cv_mae_kept']:6.0%}   {r['rain_effect']:6.2f}"
        )
    perfect_rain = float(np.exp(dict(zip(refs["perfect"]["model"].names, refs["perfect"]["model"].raw_effects()))["rain_vs_lags"]))
    print(f"   ('kept' = share of the no-weather → perfect gain that survives; perfect weather's rain × is {perfect_rain:.2f})")

    # one season of typical forecasts, checked day by day against the no-weather model
    typical = [r for r in runs if r.scenario == "typical forecast" and r.seed == 0]
    val_idx = refs["no weather"]["val_pred"].index
    y = grid.loc[val_idx, TARGET]
    print("\n4) typical forecast, draw 0, against the no-weather model (day-block bootstrap, validation)")
    for r in typical:
        d = day_block_bootstrap(y, r.val_pred.reindex(val_idx), refs["no weather"]["val_pred"], grid.loc[val_idx, "date"])
        print(f"   {r.mode:<22} MAE {d['diff']:+5.1f}   95% [{d['lo']:+5.1f}, {d['hi']:+5.1f}]")

    out = HERE / "output"
    order = [sc.name for sc in SCENARIOS]
    week = grid.index.isin(val_idx) & (grid["date"] >= SAMPLE_WEEK[0]) & (grid["date"] <= SAMPLE_WEEK[1])
    by_mode = {r.mode: r.val_pred for r in typical}
    paths = [
        StressPlots(out).survival(summary, refs, order),
        StressPlots(out).rain_trust(summary, perfect_rain, order),
        BikePlots(out).sample_week(
            grid.loc[week, "ts"],
            grid.loc[week, TARGET],
            {
                "no weather": refs["no weather"]["val_pred"].reindex(grid.index[week]),
                **{f"typical forecast, {m}": by_mode[m].reindex(grid.index[week]) for m in MODES},
            },
            title="the rainy Friday again, with a typical (simulated) forecast",
        ),
    ]
    print("\n5) plots saved")
    for p in paths:
        print("  ", p.relative_to(HERE))
    print("\nDay 88 done.")


if __name__ == "__main__":
    run_pipeline()
