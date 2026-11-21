# Presentation — Transit Recovery Bench

1. I added exopipe/bench/bench.py inside the existing exopipe package.
2. I read cached TESS observations of π Men and injected periodic dips into identical cadences for each method.
3. I compared raw, flattened and median-filtered signals with one Astropy BLS grid.
4. The first grid (0.1–1% depth) gave 9/9 for every method, which told me nothing. So I extended it down to 50 ppm.
5. Result: the default flatten loses all 200 ppm transits that the robust median keeps (16/21 vs 12/21 overall).

**Main limitation.** One quiet star, three periods per depth. These are directional results, not a completeness curve.
