# 100-Day Data Science Challenge: From Scratch to Job-Ready 🚀

A day-by-day record of learning data science, from Python basics to two end-to-end projects. From Day 36 on, most days are a small, tested NumPy/pandas module written from scratch, plus a runnable walkthrough and notes on what the numbers showed.

- **Where things stand:** [`PROGRESS.md`](PROGRESS.md) (the daily log)
- **Every day, with links:** [`INDEX.md`](INDEX.md) (generated)
- **What the two projects taught:** [`LESSONS.md`](LESSONS.md)

---

## 🧪 Projects

- **[Project 1 — Who should a retention team call?](phase_1_python_programming/day-85-churn-final-test/README.md)** (Days 81–85). Telco churn with a from-scratch logistic and boosted trees, a frozen split, and a call list sized by expected profit. On the one-time test slice: AUC 0.841, and $4,181 more than calling a fixed top 20%.
- **[Project 2 — How many bikes will be rented each hour tomorrow?](phase_1_python_programming/day-89-bike-final-test/README.md)** (Days 86–89). A day-ahead hourly forecast for Capital Bikeshare: a time split, a ridge correction to a 4-week baseline, and weather from simulated forecasts. On the one-time test quarter: MAE 46.9 bikes per hour against 60.6 for the baseline. Forecast weather added under 1 bike an hour, which isn't proven.

---

## ⚡ Quick start

Developed on Python 3.12 with NumPy, pandas, and matplotlib (a few Phase 2 days also use seaborn, plotly, polars, pyarrow, and duckdb).

```bash
python3 tools/run_day.py --list               # every day, and whether it has a walkthrough and tests
python3 tools/run_day.py 87                   # run Day 87's walkthrough (plots land in its output/)
python3 tools/run_day.py 87 --tests           # run Day 87's tests
python3 tools/run_day.py 85 -- --rehearse     # pass arguments through to a day's main.py
python3 tools/run_day.py 91                   # one validation day's bike forecast, as JSON
python3 tools/run_day.py 91 -- --serve        # then listen on http://127.0.0.1:8765
python3 tools/check_all.py                    # every test suite, as a health report
python3 tools/build_index.py                  # regenerate INDEX.md after adding a day
```

Project data is downloaded once and checked by sha256: run Day 81 (churn) or Day 86 (bikes) first.

---

## 📂 Workspace Structure

```
.
├── README.md                      # this page
├── PROGRESS.md                    # the daily log and phase checklist
├── INDEX.md                       # generated: every day, grouped by phase, with links
├── LESSONS.md                     # what Projects 1 and 2 taught
├── phase_1_python_programming/    # one folder per day: day-NN-short-name
│   ├── day-01-setup/
│   ├── ...
│   ├── day-89-bike-final-test/    # typical later day: *_engine.py, visualizer.py,
│   │                              #   main.py, test_*.py, *_NOTES.md, output/ (ignored)
│   └── day-91-bike-forecast-api/  # that forecast over HTTP; the test quarter stays closed
├── tools/                         # run_day.py, check_all.py, build_index.py, daycatalog.py
├── dashboard/                     # local progress tracker (HTML / CSS / JS)
└── templates/                     # daily notebook and script templates
```

Every day folder follows the same `day-NN-short-name` pattern (Day 90 renamed the early ones), so paths work unquoted and `tools/` can find any day by number.

---

## 📅 How the 100 days are organized

| phase | days | focus |
|---|---|---|
| Phase 1 | 1–15 | Python and NumPy fundamentals |
| Phase 2 | 16–35 | pandas, visualization, fast data tools (Parquet, DuckDB, Polars) |
| Phase 3 | 36–60 | linear algebra, calculus, statistics, and classic ML from scratch |
| Phase 4 | 61–80 | the ML workflow: leakage, scaling, tuning, calibration, optimizers, regularization |
| Phase 5 | 81–100 | projects and portfolio |

This is the plan `PROGRESS.md` follows. The dashboard's `curriculum.json` still carries the original seven-phase syllabus, so some of its later day titles differ from what the repo did (its Day 80, for example, is CNNs, where the repo's Day 80 was the Phase 4 wrap-up).

---

## 📊 The Interactive Dashboard

A small HTML/CSS/JavaScript tracker in `dashboard/`: a progress circle, daily checklists, a streak counter, auto-saved study notes (browser local storage, with a Markdown preview), and search and filters by topic or status.

Serve it from the repo root so the browser can load `curriculum.json`:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/dashboard/`. Opening `dashboard/index.html` directly also works in most browsers.

---

## 💡 Daily Routine

1. Check the day's topic in `PROGRESS.md` (or the dashboard).
2. Create `phase_1_python_programming/day-NN-short-name/`, starting from `templates/daily_template.py` or `.ipynb`.
3. Write the code, a `main.py` walkthrough, `test_*.py`, and a short notes file with the real numbers.
4. Run `python3 tools/run_day.py NN --tests`, update `PROGRESS.md`, and regenerate the index with `python3 tools/build_index.py`.

Good luck, Dinesh! Keep learning, keep building, keep committing.
