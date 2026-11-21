1. I added exopipe/bench/bench.py inside the existing exopipe package.
2. I read cached TESS observations and injected periodic dips into identical cadences for each method.
3. I compared raw, flattened and median-filtered signals using Astropy BLS.
4. I tested phase wrapping, short curves and strong-signal recovery in test_transit.py.
5. I exported recovered periods, errors, timing and the FITS checksum to recovery.csv.

**Main limitation.** One target and period agreement alone cannot establish population-level completeness or a calibrated false-positive rate.
