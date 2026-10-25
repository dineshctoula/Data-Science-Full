# Day 84 — From probabilities to a call list

Project 1, step 4. Days 81–83 asked which model ranks leavers best, and Day 82's logistic held its place. Today asks the question a retention team actually has: **who should we call, and how many?** Test is still closed.

## The assumptions (placeholders, not data)

| item | value |
|---|---|
| cost of a call | $10 agent time + $20 credit = **$30**, paid for everyone called |
| save rate | **0.25**: a called leaver stays a quarter of the time |
| value of a save | 12 months × 50% margin × monthly bill |

A call pays when **p > cost / (save rate × value)**. The cut-off depends on the bill:

| monthly bill | break-even churn probability |
|---:|---:|
| $20 | 1.00 (never worth a call) |
| $50 | 0.40 |
| $80 | 0.25 |
| $110 | 0.18 |

![break-even map](output/break_even_map.png)

The big cluster at $20 is customers with no internet service. However risky they look, a $30 call can't pay for itself on a $20 bill at this save rate.

## Six lists on the 1056 validation customers

| list | calls | leavers reached | expected saves | cost | profit |
|---|---:|---:|---:|---:|---:|
| call everyone | 1056 | 280 | 70.0 | $31,680 | $810 |
| top 20% by risk | 211 | 155 | 38.8 | $6,330 | $12,317 |
| top 20% by expected value | 211 | 154 | 38.5 | $6,330 | $13,604 |
| **break-even rule** | **406** | 225 | 56.2 | $12,180 | **$16,244** |
| break-even, overconfident p (log-odds × 2) | 317 | 205 | 51.2 | $9,510 | $16,035 |
| break-even, underconfident p (log-odds × 0.5) | 539 | 244 | 61.0 | $16,170 | $14,412 |
| *hindsight best size (peeks at outcomes)* | *363* | | | | *$16,699* |

- **Rank by value, not just risk.** At the same 211 calls, ranking by p × value instead of p earns $1,300 more by reaching the bigger bills.
- **Let the economics size the list.** The break-even rule calls 406 customers, almost twice the fixed 20%. It earns **$3,927 more** than the top 20% by risk, 95% paired-bootstrap interval [$2,045, $5,813].
- **It needs no tuning.** The rule's cut-off comes from the assumptions, not from validation outcomes, and it still reaches **97% of the hindsight ceiling**. The hindsight list size uses outcomes the team won't have, so it's a ceiling and not a strategy.
- **Calibration matters, but the peak is flat.** The rule trusts the probabilities, so it only works if they're honest. That's why Days 82–83 kept checking log loss. Overconfident probabilities shrank the list to 317 and cost about $200. Underconfident ones grew it to 539 and cost about $1,800. The profit curve is flat near its peak, so moderate miscalibration is forgiving, but not free.

![profit curves](output/profit_curves.png)

## The save rate nobody has measured

| save rate | rule calls | rule profit | top-20% profit | hindsight |
|---:|---:|---:|---:|---:|
| 0.05 | 0 | $0 | −$2,601 | $0 |
| 0.10 | 148 | $1,671 | $1,129 | $1,821 |
| 0.15 | 279 | $5,955 | $4,858 | $6,168 |
| 0.20 | 337 | $11,164 | $8,588 | $11,344 |
| **0.25** | **406** | **$16,244** | **$12,317** | **$16,699** |
| 0.30 | 444 | $21,945 | $16,047 | $22,216 |
| 0.40 | 496 | $33,618 | $23,506 | $33,849 |
| 0.50 | 541 | $45,196 | $30,964 | $45,906 |

![sensitivity](output/sensitivity.png)

- At a **5% save rate** no call pays. The rule calls nobody, while the fixed 20% list loses $2,600.
- From 0.10 up, the rule stays within a few percent of the hindsight ceiling. The fixed list falls further behind as the save rate rises, because it never grows.
- The save rate changes the right list size more than anything else. Before running a campaign, the most valuable thing to measure is the save rate itself, for example with a small randomized trial of called vs not-called at-risk customers.

## Takeaway

The headline metric from Day 81 (top-20% recall) was a stand-in. Once costs and values are written down, the best list isn't a fixed 20%. It's everyone whose expected profit is positive, and that only works because the logistic's probabilities are calibrated. **The project model is now the logistic plus the break-even rule.** Next: open the test slice once and score both.
