# Sub-Minute Decay Report

## Design

- Bars: every integer width from 1s to 30s.
- Schemes: `best`, `sum`, `distance`, `pca`.
- Feature: single current-bar OFI.
- Horizon: 1 bar.
- Estimator: tuned ridge with fixed UTC 7-day train / 1-day test / 7-day step.
- Controls: own OFI, cross-return history and exact day-shifted placebo.
- Correction: Bonferroni/Holm/BH with `m = 792`.

## Corrected Pass Counts

| Test | Passing cells after Bonferroni(792) |
|---|---:|
| Cross beats own OFI | 34 / 120 |
| Cross beats cross-return history | 21 / 120 |
| Real cross beats shifted placebo | 34 / 120 |
| Passes all three controls | 21 / 120 |

## Boundary

All four schemes pass all three controls through 5s. The first bar width where
all four schemes stop passing is 6s; 6s has one all-control passing scheme. No
later bar width is treated as a clean continuation of the effect.

Some non-divisor bar widths have structurally non-computable placebo summaries
because the exact one-day shifted timestamp is absent from the 80%-coverage
filtered bar-end clock. These rows remain in `results.csv` and
`significance.csv`; the report treats them transparently rather than removing
them after the fact.
