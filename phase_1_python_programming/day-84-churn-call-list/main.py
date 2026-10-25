"""Run the Day 84 call-list walkthrough."""

import numpy as np

from project_paths import DATA_DIR, HERE, SPLIT_FILE
from churn_data import TARGET, load_clean
from economics import Economics
from policy_engine import (
    bootstrap_profit_diff,
    call_if_profitable,
    call_top_frac,
    champion_scores,
    evaluate,
    hindsight_best,
    profit_curve,
    sensitivity,
    sharpen,
)
from split_engine import load_split
from visualizer import CallPlots

SAVE_RATES = np.round(np.arange(0.05, 0.501, 0.05), 2)


def _row(name, r):
    print(
        f"   {name:<32} {r['calls']:5d}  {r['leavers_reached']:7d}  {r['expected_saves']:6.1f}  "
        f"${r['cost']:>8,.0f}  ${r['profit']:>8,.0f}"
    )


def run_pipeline():
    print("=" * 72)
    print("DAY 84: FROM PROBABILITIES TO A CALL LIST")
    print("=" * 72)

    econ = Economics()
    parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
    train, val = parts["train"], parts["validation"]
    y = val[TARGET].to_numpy()
    print("\n1) assumptions (placeholders; swap in real ones)")
    print(f"   each call: ${econ.call_cost:g} agent time + ${econ.credit:g} credit = ${econ.cost:g}")
    print(f"   a called leaver stays with probability {econ.save_rate:g}")
    print(f"   a saved customer is worth {econ.months_kept:g} months × {econ.margin:.0%} margin × monthly bill")
    value = econ.value(val)
    cut = econ.break_even(val)
    print("   so a call pays when churn probability > cost / (save rate × value):")
    for bill in (20, 50, 80, 110):
        b = econ.cost / (econ.save_rate * bill * econ.months_kept * econ.margin)
        print(f"     ${bill:>3}/month -> break-even probability {min(b, 1):.2f}{'  (never worth a call)' if b >= 1 else ''}")

    # the champion from Day 82, refit on train; validation outcomes are only used to score
    p = champion_scores(train, val)
    print(f"\n2) Day 82 logistic on the {len(val)} validation customers ({int(y.sum())} leavers)")

    lists = {
        "call everyone": np.ones(len(y), dtype=bool),
        "top 20% by risk": call_top_frac(p, 0.20),
        "top 20% by expected value": call_top_frac(p * value, 0.20),
        "break-even rule": call_if_profitable(p, val, econ),
        "break-even, overconfident p": call_if_profitable(sharpen(p, 2.0), val, econ),
        "break-even, underconfident p": call_if_profitable(sharpen(p, 0.5), val, econ),
    }
    print("\n3) each list scored on validation")
    print(f"   {'list':<32} {'calls':>5}  {'leavers':>7}  {'saves':>6}  {'cost':>9}  {'profit':>9}")
    results = {name: evaluate(called, y, val, econ) for name, called in lists.items()}
    for name, r in results.items():
        _row(name, r)
    k_best, best = hindsight_best(p * value, y, val, econ)
    print(f"   {'hindsight best size (peeks at y)':<32} {k_best:5d}  {'':>7}  {'':>6}  {'':>9}  ${best:>8,.0f}")

    diff = bootstrap_profit_diff(lists["break-even rule"], lists["top 20% by risk"], y, val, econ)
    print("\n4) break-even rule minus top 20% by risk (paired bootstrap over customers)")
    print(f"   ${diff['diff']:,.0f}   95% [${diff['lo']:,.0f}, ${diff['hi']:,.0f}]   above zero {diff['share_above_zero']:.0%}")
    print(f"   the rule reaches {results['break-even rule']['profit'] / best:.0%} of the hindsight ceiling")

    table = sensitivity(p, y, val, econ, SAVE_RATES)
    print("\n5) how the answer moves with the save rate")
    print(f"   {'save rate':>9}  {'rule calls':>10}  {'rule profit':>11}  {'top-20% profit':>14}  {'hindsight':>10}")
    for _, r in table.iterrows():
        print(
            f"   {r['save_rate']:9.2f}  {int(r['rule_calls']):10d}  ${r['rule_profit']:>10,.0f}  "
            f"${r['top_profit']:>13,.0f}  ${r['hindsight_profit']:>9,.0f}"
        )

    order_risk = np.argsort(-p, kind="stable")
    order_value = np.argsort(-(p * value), kind="stable")
    rule = results["break-even rule"]
    top = results["top 20% by risk"]
    plots = CallPlots(HERE / "output")
    paths = [
        plots.profit_curves(
            {
                "ranked by expected value (p × value)": profit_curve(order_value, y, val, econ),
                "ranked by risk (p)": profit_curve(order_risk, y, val, econ),
            },
            [
                ("top 20% by risk", top["calls"], top["profit"]),
                ("break-even rule", rule["calls"], rule["profit"]),
            ],
        ),
        plots.break_even_map(val["MonthlyCharges"], p, np.minimum(cut, 1.0), lists["break-even rule"]),
        plots.sensitivity(table, econ.save_rate),
    ]
    print("\n6) plots saved")
    for path in paths:
        print("  ", path.relative_to(HERE))
    print("\nDay 84 done. Test is still closed.")


if __name__ == "__main__":
    run_pipeline()
