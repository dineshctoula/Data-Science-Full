"""Run the Day 63 categorical-encoding walkthrough."""

from encode_engine import OneHotEncoder, OrdinalEncoder, compare_encodings, make_plan_data
from visualizer import EncodePlots


def run_pipeline():
    print("=" * 72)
    print("DAY 63: CATEGORICAL ENCODING")
    print("=" * 72)

    plan, channel, hours, score = make_plan_data(n=200, seed=63)

    print("\n1) mean score by plan (this one really is ordered)")
    for name in ("basic", "plus", "pro"):
        print(f"   {name:<8} {score[plan == name].mean():6.1f}")

    print("\n2) mean score by channel (no natural order)")
    for name in ("email", "ads", "referral"):
        print(f"   {name:<10} {score[channel == name].mean():6.1f}")

    right = OrdinalEncoder(order=("basic", "plus", "pro")).fit(plan)
    print("\n3) ordinal codes:", dict(zip(right.categories_, range(len(right.categories_)))))
    print("   an unseen plan 'enterprise' ->", int(right.transform(["enterprise"])[0]))

    oh = OneHotEncoder().fit(channel)
    sample = oh.transform(["referral", "sms"])
    print("\n4) one-hot channel columns:", oh.categories_)
    print("   referral row:", sample[0].astype(int).tolist())
    print("   unseen 'sms':", sample[1].astype(int).tolist(), "(all zeros)")

    print("\n5) same holdout, three encodings")
    rows = compare_encodings(plan, channel, hours, score, seed=63)
    for row in rows:
        print("  ", row.summary())

    plots = EncodePlots()
    p1 = plots.mean_bars(plan, score, title="mean score by plan", filename="plan_means.png")
    p2 = plots.mean_bars(channel, score, title="mean score by channel", filename="channel_means.png")
    p3 = plots.mse_bars(rows)

    print("\n6) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 63 done.")


if __name__ == "__main__":
    run_pipeline()
