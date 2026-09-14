# Day 40 — Descriptive Statistics

## Deliverables

- `stats_engine.py`: mean, median, modes, sample vs population variance, linear percentiles, Tukey fences, z-scores, and moment skewness.
- `visualizer.py`: a histogram with mean/median markers, a five-number box plot, and a z-score strip.
- `main.py`: a runnable lesson that checks a symmetric textbook sample, then summarizes simulated exam scores.
- Two test modules covering hand-check identities, outlier fences, z-score scaling, and plot output.

## Key results

On `[1, 2, 3, 4, 5]` the mean and median are both `3`. Population variance is `2` (`÷ n`); sample variance is `2.5` (`÷ n − 1`). Quartiles are `Q1 = 2` and `Q3 = 4`. Skewness is `0` because the tails match.

The exam-score simulation (`n = 80`) is slightly right-skewed: mean `67.948` sits to the right of median `64.515`, and moment skewness is `0.139`. The five-number summary is `18.00, 60.18, 64.51, 74.21, 100.00`. The 1.5-IQR fences `[39.14, 95.25]` flag five outliers, including the low score `18` (`z = −3.89`).

## Reading the formulas

The comments in `stats_engine.py` mark the distinctions to look for while studying a sample:

1. The mean uses every value; the median uses only order. A right tail pulls the mean above the median.
2. Sample variance divides by `n − 1` so one sample does not systematically understate spread.
3. Linear percentiles use rank `p / 100 × (n − 1)`, which is why `Q1` and `Q3` on the textbook list are exactly `2` and `4`.
4. Tukey fences `Q1 − 1.5 IQR` and `Q3 + 1.5 IQR` flag unusual points without assuming a Normal curve.
5. A z-score is `(x − mean) / sample std`; the transformed sample has mean `0` and sample standard deviation `1`.

## Practical checks

- Empty, two-dimensional, and non-finite inputs are rejected before a summary is computed.
- Z-scores and skewness are undefined when every observation is the same.
- All-unique samples return no mode instead of treating every value as a mode.
- A logarithmic scale is not used on the histogram, because counts—not losses—are being compared.

## Run

```bash
cd phase_1_python_programming/day-40-descriptive-statistics
python3 -m unittest -v
python3 main.py
```

The pipeline writes `output/distribution_histogram.png`, `output/five_number_boxplot.png`, and `output/zscore_strip.png`.
