# Sub-Minute Temporal Stability Check

This check evaluates the already-selected 1s-6s sub-minute region on one
calendar split inside the same historical sample. It is not a fresh holdout.

## Design

- Frozen cells: `bar_s in {1,2,3,4,5,6}` x `{best,sum,distance,pca}`.
- Feature: single current-bar OFI.
- Horizon: 1 bar.
- Train window: first 6 calendar weeks.
- Evaluation window: final 2 calendar weeks.
- Controls: own OFI, cross-return history and exact day-shifted placebo.

## Result

All 24 frozen cells pass the three raw reference controls on this split. This
supports temporal stability inside the observed sample, but it does not replace
a fresh-data holdout test.
