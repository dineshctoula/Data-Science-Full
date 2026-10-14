# Day 74 — Mini-batch gradient descent

240 rows, same line as yesterday: `y = 3 x0 - 1.5 x1` plus noise. Learning rate fixed at 0.2. 168 training rows, 72 held out. An epoch is one shuffle of those 168 rows. The batch size changes how many steps happen inside that shuffle.

| batch | updates per epoch | test MSE epoch 1 | test MSE epoch 5 | test MSE epoch 20 |
|---:|---:|---:|---:|---:|
| 168 (full) | 1 | 10.835 | 2.817 | 0.861 |
| 32 | 6 | 2.154 | 0.843 | 0.857 |
| 1 | 168 | 0.816 | 1.418 | 1.087 |

168 / 32 is 5.25, so the size-32 run takes 5 full chunks and a leftover of 8. That is 6 updates per pass. Full batch takes 1.

After one pass, batch 1 looks best (test MSE **0.816**) because it has already stepped 168 times. By epoch 5 that same run is at **1.418**. It never sits still. At epoch 20 it is **1.087**, a bit worse than the other two.

Full batch and batch 32 finish almost together: **0.861** and **0.857**. The size-32 run was already there around epoch 5. Full batch needed the rest of the 20 passes to catch up.

## Takeaway

Count updates, not just epochs. A smaller batch means more steps per pass and a jumpier curve. Here the one-row updates arrived first and then wandered. Batch 32 and the full table ended in the same place.
