# Day 57 — Cross-validation

Nearest-centroid on two overlapping clouds (seed=57, n=100).

## Splits

- plain k-fold: shuffle, chop into k, rotate which chunk is the test set
- stratified: do that *inside* each class so a fold doesn't end up all one label

Every row lands in exactly one test fold.

## Numbers

**Stratified 5-fold:** mean **0.900**, std 0.087  
folds: 0.75, 0.95, 0.95, 0.90, 0.95

**Plain 5-fold:** mean **0.880**, std 0.067

**8 random 30% holdouts:** mean 0.883, std 0.044

Holdout std came out *smaller* this seed. Each CV test fold is only 20 rows,
so one miss moves accuracy by 0.05 — the std looks big even when the model
is fine. The holdouts are 30 rows and this cloud is easy, so they didn't
bounce much. Point of the exercise is still: report the mean and the spread,
not a single split you happened to like.

**k sweep** (stratified means): 2→0.90, 3→0.88, 5→0.90, 8→0.91, 10→0.89.
k barely moved the average. 5 is the usual default and it was fine here.

## Takeaways

- Fresh model every fold. Don't train once and slice the predictions.
- Stratify when classes are uneven.
- Small test folds = noisy per-fold scores. Trust the mean more than fold 1.
