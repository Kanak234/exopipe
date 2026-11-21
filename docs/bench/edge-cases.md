# Tested edge cases
- phase wrap (box centred on the epoch wraps across period boundaries)
- short time baseline (< 101 points or < 10 days rejected)
- deterministic strong-transit recovery (1% depth always recovered on a flat curve)
- row count = periods × depths × methods (guards the DEPTHS grid)
- null controls (depth 0 rows present, excluded from counts)
