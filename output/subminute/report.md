# Sub-Minute Robustness Report

## Design

- Bars: 1s, 5s and 10s.
- Schemes: `best`, `sum`, `distance`, `pca`.
- Feature: single current-bar OFI.
- Horizon: 1 bar.
- Estimator: tuned ridge with lambdas `{0,10,100}`.
- Walk-forward: fixed UTC 7-day train / 1-day test / 7-day step.
- Controls: own OFI, cross-return history and exact day-shifted cross-feature
  placebo.

## Corrected Significance

| Test | Passing cells after Bonferroni(684) |
|---|---:|
| Cross beats own OFI | 10 / 12 |
| Cross beats cross-return history | 8 / 12 |
| Real cross beats shifted placebo | 10 / 12 |

## Interpretation

The 1s and 5s cells pass all three controls across all four OFI schemes. The
10s cells remain positive against own OFI and placebo in several cases, but do
not beat cross-return history after correction. The supported sub-minute result
is therefore concentrated at 1s-5s bars in this sample.
