"""Run the Day 85 final test — or rehearse it on validation first.

    python3 main.py --rehearse   fit on train, score validation (test untouched)
    python3 main.py              fit on train + validation, open test once
"""

import argparse

import numpy as np
import pandas as pd

from project_paths import CALL_LIST, DATA_DIR, FIRST_MODEL, HERE, KICKOFF, SPLIT_FILE, load_from
import final_recipe as R
from churn_data import TARGET, load_clean
from final_engine import Baseline, fit_champion, policy_card, score_card
from holdout_gate import fingerprint, open_test
from split_engine import load_split
from visualizer import FinalPlots

LOCK = HERE / "test_opened.lock"
# the lock covers the recipe and the code that turns it into numbers
RECIPE_FILES = (HERE / "final_recipe.py", HERE / "final_engine.py")


def _card_line(name, card):
    parts = [f"{name:<30}"]
    for key in ("top20_recall", "auc", "log_loss"):
        point, lo, hi = card[key]
        parts.append(f"{point:.3f} [{lo:.3f}, {hi:.3f}]")
    print("   " + "  ".join(parts))


def run_pipeline(rehearse: bool = False):
    print("=" * 72)
    print("DAY 85: THE FINAL TEST" + ("  (REHEARSAL ON VALIDATION)" if rehearse else ""))
    print("=" * 72)

    parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
    train, val = parts["train"], parts["validation"]
    recipe_fp = fingerprint(RECIPE_FILES)
    fit_desc = "train only (rehearsal)" if rehearse else " + ".join(R.FIT_ON)
    print(f"\n1) recipe {recipe_fp[:12]}…: {R.VARIANT}, lam {R.LAM:g}, fit on {fit_desc}")

    if rehearse:
        # a dress rehearsal: same steps, but validation plays the part of test
        fit_df, held, held_name = train, val, "validation"
        out = HERE / "output" / "rehearsal"
    else:
        held, first_time = open_test(parts, LOCK, recipe_fp)
        print(f"   test opened {'for the first time; lock written' if first_time else 'again with the identical recipe'}")
        # every choice is made, so validation joins the fit rows
        fit_df = pd.concat([parts[name] for name in R.FIT_ON], ignore_index=True)
        held_name = "test"
        out = HERE / "output"
    y = held[TARGET].to_numpy()
    print(f"   fit rows {len(fit_df)}, {held_name} rows {len(held)} ({int(y.sum())} leavers)")

    champion = fit_champion(fit_df, R.VARIANT, R.LAM)
    p = champion.predict(held)
    base_p = Baseline(R.BASELINE_KEYS).fit(fit_df).predict(held)

    cards = {}
    # the validation card is the number Days 82-84 reported: train-only fit
    p_val = fit_champion(train, R.VARIANT, R.LAM).predict(val)
    cards["validation, train fit"] = score_card(val[TARGET], p_val, R.N_BOOT, R.SEED)
    if not rehearse:
        p_train_only = fit_champion(train, R.VARIANT, R.LAM).predict(held)
        cards["test, train fit"] = score_card(y, p_train_only, R.N_BOOT, R.SEED)
    cards[f"{held_name}, final fit"] = score_card(y, p, R.N_BOOT, R.SEED)
    cards[f"{held_name}, lookup floor"] = score_card(y, base_p, R.N_BOOT, R.SEED)

    print("\n2) score cards (95% bootstrap intervals)")
    print(f"   {'':<30}  {'top-20% recall':<21}  {'AUC':<21}  {'log loss':<21}")
    for name, card in cards.items():
        _card_line(name, card)
    final = cards[f"{held_name}, final fit"]
    print(f"   mean predicted churn {final['mean_p']:.3f} vs actual {final['churn_rate']:.3f}")

    econ = R.ECONOMICS
    pc = policy_card(p, y, held, econ, R.TOP_FRAC, R.N_BOOT, R.SEED)
    print(f"\n3) call lists in dollars on {held_name} (Day 84 economics, save rate {econ.save_rate:g})")
    for name, key in (("break-even rule", "rule"), (f"top {R.TOP_FRAC:.0%} by risk", "fixed")):
        r = pc[key]
        print(f"   {name:<20} {r['calls']:4d} calls, {r['leavers_reached']:3d} leavers, profit ${r['profit']:>8,.0f}")
    g = pc["gap"]
    print(f"   rule minus fixed: ${g['diff']:,.0f}  95% [${g['lo']:,.0f}, ${g['hi']:,.0f}]")
    print(f"   hindsight ceiling ${pc['ceiling']:,.0f}; the rule reaches {pc['rule']['profit'] / pc['ceiling']:.0%}")

    KickoffPlots = load_from(KICKOFF, "visualizer").KickoffPlots
    ModelPlots = load_from(FIRST_MODEL, "visualizer").ModelPlots
    CallPlots = load_from(CALL_LIST, "visualizer").CallPlots
    value = econ.value(held)
    from policy_engine import profit_curve

    paths = [
        FinalPlots(out).val_vs_test(cards),
        KickoffPlots(out).gains(
            y,
            {"Day 81 lookup": base_p, "final logistic": p, "perfect list": y.astype(float)},
            frac=R.TOP_FRAC,
            title=f"{held_name} gains: final logistic vs the lookup floor",
        ),
        ModelPlots(out).calibration(y, p, title=f"{held_name}: predicted vs actual churn"),
        CallPlots(out).profit_curves(
            {
                "ranked by expected value (p × value)": profit_curve(np.argsort(-(p * value), kind="stable"), y, held, econ),
                "ranked by risk (p)": profit_curve(np.argsort(-p, kind="stable"), y, held, econ),
            },
            [
                (f"top {R.TOP_FRAC:.0%} by risk", pc["fixed"]["calls"], pc["fixed"]["profit"]),
                ("break-even rule", pc["rule"]["calls"], pc["rule"]["profit"]),
            ],
            title=f"{held_name}: profit as the call list grows",
            ylabel=f"profit on {held_name} ($)",
        ),
    ]
    print("\n4) plots saved")
    for path in paths:
        print("  ", path.relative_to(HERE))
    print("\nRehearsal done; test untouched." if rehearse else "\nDay 85 done. Project 1's test slice has been used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rehearse", action="store_true", help="score validation instead of opening test")
    run_pipeline(rehearse=parser.parse_args().rehearse)
