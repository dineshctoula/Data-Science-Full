# Day 77 — Learning-rate schedules

240 rows of `y = 3 x0 - 1.5 x1` plus noise. Every run starts at learning rate **1.8** for 60 full-batch epochs. That rate is past the comfortable range from Day 73.

| epoch | constant lr / test | step decay lr / test | cosine lr / test |
|---:|---|---|---|
| 1 | 1.80 / 6.290 | 1.80 / 6.290 | 1.80 / 6.290 |
| 15 | 1.80 / 1.552 | 1.80 / 1.552 | 1.56 / 0.798 |
| 20 | 1.80 / 2.082 | 0.90 / 0.814 | 1.38 / 0.819 |
| 60 | 1.80 / **1.741** | 0.225 / **0.814** | 0.00 / **0.814** |

Left alone, the constant rate is still bouncing at epoch 60 (test MSE **1.741**). Step decay halves every 15 epochs; after the first cut at epoch 16 the test error drops to **0.814** and stays there. Cosine cools smoothly and lands on the same **0.814**.

## Takeaway

A rate that is too high for the end of training can still be useful at the start if you plan to shrink it. On this table the schedule is what rescues the run. Cosine and step decay finish in the same place.
