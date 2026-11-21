# PRD — Transit Recovery Bench (exopipe module)

## Problem
exopipe detrends every light curve before its BLS search. Detrending removes slow stellar variability, but it can also eat part of a real transit. Without a controlled test, there is no evidence for which preprocessing to use, or at what depth it starts losing planets.

## Users
- Kanak and other exopipe contributors choosing the default preprocessing.
- Reviewers who want a reproducible number behind that choice.

## Goals
1. Measure period recovery for raw, flattened and robust-median curves on identical injected transits.
2. Cover the depth range where the methods actually differ (50 ppm to 1%).
3. Be fully reproducible offline: cached FITS with checksum, fixed seeds, CSV of every trial.

## Non-goals
- Population completeness or false-alarm calibration (that needs many stars).
- Replacing exopipe's classifier evaluation.

## Acceptance criteria
- `python -m pytest -q exopipe/bench` passes.
- `python -m exopipe.bench.bench data/bench/tic261136679-sector01.fits recovery.csv` prints per-method totals and a per-depth table, and writes one CSV row per trial (3 periods × 8 depths × 3 methods = 72).
- At least one depth separates the methods (no ceiling). Currently 200 ppm does.
