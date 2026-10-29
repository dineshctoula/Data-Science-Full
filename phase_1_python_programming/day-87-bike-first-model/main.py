"""Run the Day 87 first bike-demand model."""

import numpy as np

from project_paths import DATA_DIR, HERE, SPLIT_FILE, load_kickoff
from baseline_engine import day_block_bootstrap, mae, wape
from bike_data import TARGET, load_grid
from evaluate_engine import baseline_cv, error_by, score_ladder
from features import HUM_KNOTS, TEMP_KNOTS, offset
from split_engine import assign_slices, load_split_dates, scorable
from visualizer import ModelPlots

BikePlots = load_kickoff("visualizer").BikePlots
SAMPLE_WEEK = ("2012-08-06", "2012-08-12")
WEATHER_NAMES = {1: "clear", 2: "mist / cloudy", 3: "rain / snow", 4: "heavy rain"}


def _hinge_curve(model, names, prefix, knots, xs):
    """f(x) = a·x + Σ b_k·max(0, x − k), from the raw slopes of one hinge block."""
    effects = dict(zip(names, model.raw_effects()))
    f = effects[f"{prefix}_vs_lags"] * xs
    for k in knots:
        f = f + effects[f"{prefix}_over_{k:g}_vs_lags"] * np.maximum(0.0, xs - k)
    return f


def run_pipeline():
    print("=" * 72)
    print("DAY 87: FIRST BIKE-DEMAND MODEL")
    print("=" * 72)

    grid = load_grid(DATA_DIR)
    slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
    base = offset(grid)
    val = (slices == "validation") & scorable(grid) & base.notna()
    y = grid.loc[val, TARGET]
    base_val = mae(y, base[val])
    base_cv = baseline_cv(grid, slices)
    print("\n1) frozen Day 86 split; test stays closed")
    print(f"   bar: 4-week same-hour mean, validation MAE {base_val:.1f}, rolling CV MAE {np.mean(list(base_cv.values())):.1f}")

    results = score_ladder(grid, slices)
    print("\n2) ridge on log1p(y) − log1p(baseline); penalty picked by rolling CV (Jan–Jun 2012)")
    print(f"   {'rung':<38} {'cols':>4} {'lam':>5}  {'CV MAE':>6}  {'val MAE':>7}  {'val WAPE':>8}")
    print(f"   {'4-week baseline':<38} {'':>4} {'':>5}  {np.mean(list(base_cv.values())):6.1f}  {base_val:7.1f}  {wape(y, base[val]):8.1%}")
    for r in results:
        print(f"   {r.rung:<38} {r.n_features:4d} {r.lam:5g}  {r.cv_mae:6.1f}  {r.val_mae:7.1f}  {wape(y, r.val_pred):8.1%}")

    shape, honest, upper = results
    print("\n3) validation gaps, day-block bootstrap (92 days)")
    for label, a, b in (
        ("no-weather model − baseline", honest.val_pred, base[val]),
        ("weather model − baseline", upper.val_pred, base[val]),
        ("weather model − no-weather model", upper.val_pred, honest.val_pred),
    ):
        d = day_block_bootstrap(y, a, b, grid.loc[val, "date"])
        print(f"   {label:<34} MAE {d['diff']:+5.1f}   95% [{d['lo']:+5.1f}, {d['hi']:+5.1f}]")

    weather = grid.loc[val, "weathersit"].astype(int)
    by_base = error_by(y, base[val], weather)
    by_model = error_by(y, upper.val_pred, weather)
    table = []
    print("\n4) validation MAE by weather")
    for (_, rb), (_, rm) in zip(by_base.iterrows(), by_model.iterrows()):
        name = WEATHER_NAMES[int(rb["group"])]
        table.append((name, int(rb["hours"]), rb["mae"], rm["mae"]))
        print(f"   {name:<14} {int(rb['hours']):5d} h   baseline {rb['mae']:6.1f}   with weather {rm['mae']:6.1f}")

    names = list(upper.model.names)
    temps = np.linspace(0.05, 0.95, 91)
    hums = np.linspace(0.2, 1.0, 81)
    # multipliers against a mild 20 °C (temp 0.5) and 60% humidity reference hour
    f_t = _hinge_curve(upper.model, names, "temp", TEMP_KNOTS, temps)
    f_h = _hinge_curve(upper.model, names, "hum", HUM_KNOTS, hums)
    t_mult = np.exp(f_t - np.interp(0.5, temps, f_t))
    h_mult = np.exp(f_h - np.interp(0.6, hums, f_h))
    print("\n5) learned weather curves (rentals vs the same hour at 20 °C / 60% humidity)")
    for c in (5, 12, 20, 29, 35):
        print(f"   {c:>2} °C -> × {np.interp(c / 41, temps, t_mult):.2f}")
    for h in (40, 60, 80, 95):
        print(f"   {h}% humidity -> × {np.interp(h / 100, hums, h_mult):.2f}")
    effects = dict(zip(names, upper.model.raw_effects()))
    print(f"   rain vs dry: × {np.exp(effects['rain_vs_lags']):.2f}; at a working-day rush hour, a further × {np.exp(effects['rain_at_peak_vs_lags']):.2f}")

    out = HERE / "output"
    plots = ModelPlots(out)
    week = val & (grid["date"] >= SAMPLE_WEEK[0]) & (grid["date"] <= SAMPLE_WEEK[1])
    paths = [
        plots.ladder([("4-week baseline", np.mean(list(base_cv.values())), base_val)] + [(r.rung, r.cv_mae, r.val_mae) for r in results]),
        plots.cv_months({"4-week baseline": base_cv, **{r.rung: r.cv_by_month for r in results}}),
        plots.weather_curves([("temperature (°C)", temps * 41, t_mult, "20 °C"), ("humidity (%)", hums * 100, h_mult, "60%")]),
        plots.error_by_weather(table),
        BikePlots(out).sample_week(
            grid.loc[week, "ts"],
            grid.loc[week, TARGET],
            {
                "4-week baseline": base[week],
                "model, no weather": honest.val_pred[week[val]].reindex(grid.index[week]),
                "model, with weather": upper.val_pred[week[val]].reindex(grid.index[week]),
            },
            title=f"validation week {SAMPLE_WEEK[0]} to {SAMPLE_WEEK[1]}: the rainy Friday",
        ),
    ]
    print("\n6) plots saved")
    for p in paths:
        print("  ", p.relative_to(HERE))
    print(f"\nDay 87 done. New bar on validation: MAE {honest.val_mae:.1f} without weather, {upper.val_mae:.1f} with it.")


if __name__ == "__main__":
    run_pipeline()
