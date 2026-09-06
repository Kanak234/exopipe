"""
catalog.py
----------
Reader for the TIC v8 CTL catalogue CSV (the big ~10 GB file from the ISRO
portal, e.g. exo_CTL_08.01xTIC_v8.1.csv).

The catalogue ships WITHOUT a header row, so we attach the official TIC v8
column names ourselves (see COLUMNS below). The file is huge, so we never
load it all into memory -- we stream it in chunks with pandas and keep only
the columns and rows we actually care about.

Why this matters: the earlier pipeline guessed column positions and ended up
reading the wrong column, which is why it kept selecting TIC IDs that had no
light curves. Here we use the documented schema, so the right columns are
read every time.
"""

import os
import pandas as pd


# Official TIC v8 CTL column order. Taken straight from the header file that
# ships alongside the catalogue. Index in this list == column index in the CSV.
COLUMNS = [
    "ID", "version", "HIP", "TYC", "UCAC", "TWOMASS", "SDSS", "ALLWISE",
    "GAIA", "APASS", "KIC", "objType", "typeSrc", "ra", "dec", "POSflag",
    "pmRA", "e_pmRA", "pmDEC", "e_pmDEC", "PMflag", "plx", "e_plx", "PARflag",
    "gallong", "gallat", "eclong", "eclat", "Bmag", "e_Bmag", "Vmag", "e_Vmag",
    "umag", "e_umag", "gmag", "e_gmag", "rmag", "e_rmag", "imag", "e_imag",
    "zmag", "e_zmag", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
    "TWOMflag", "prox", "w1mag", "e_w1mag", "w2mag", "e_w2mag", "w3mag",
    "e_w3mag", "w4mag", "e_w4mag", "GAIAmag", "e_GAIAmag", "Tmag", "e_Tmag",
    "TESSflag", "SPFlag", "Teff", "e_Teff", "logg", "e_logg", "MH", "e_MH",
    "rad", "e_rad", "mass", "e_mass", "rho", "e_rho", "lumclass", "lum",
    "e_lum", "d", "e_d", "ebv", "e_ebv", "numcont", "contratio", "disposition",
    "duplicate_id", "priority", "eneg_EBV", "epos_EBV", "EBVflag", "eneg_Mass",
    "epos_Mass", "eneg_Rad", "epos_Rad", "eneg_rho", "epos_rho", "eneg_logg",
    "epos_logg", "eneg_lum", "epos_lum", "eneg_dist", "epos_dist", "distflag",
    "eneg_Teff", "epos_Teff", "TeffFlag", "gaiabp", "e_gaiabp", "gaiarp",
    "e_gaiarp", "gaiaqflag", "starchareFlag", "VmagFlag", "BmagFlag",
    "splists", "e_RA", "e_Dec", "RA_orig", "Dec_orig", "e_RA_orig",
    "e_Dec_orig", "raddflag", "wdflag", "objID",
]

# The handful of columns we actually use downstream. Reading only these keeps
# memory and parse time low even on the full catalogue.
USE_COLUMNS = [
    "ID", "ra", "dec", "Tmag", "Teff", "logg", "rad", "mass",
    "d", "numcont", "contratio", "disposition", "priority",
]


def _column_indices(use_columns):
    """Map our wanted column names to their integer positions in the CSV."""
    return [COLUMNS.index(name) for name in use_columns]


def iter_catalog(path, chunsize=100_000, use_columns=None):
    """
    Yield the catalogue in chunks as pandas DataFrames.

    Parameters
    ----------
    path : str
        Path to the big CTL csv.
    chunize : int
        Rows per chunk. 100k is a good balance -- small enough to stay light
        on RAM, big enough that we are not spending all our time in overhead.
    use_columns : list of str or None
        Which columns to keep. Defaults to USE_COLUMNS.

    Yields
    ------
    pandas.DataFrame
        One chunk at a time, already trimmed to the wanted columns and with
        proper column names attached.
    """
    if use_columns is None:
        use_columns = USE_COLUMNS

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Catalogue not found at: {}\n".format(path)
            + "Pass the correct path with --catalog, e.g.\n"
            + "  /home/kanak07/Downloads/exo_CTL_08.01xTIC_v8.1.csv"
        )

    wanted_idx = _column_indices(use_columns)

    # header=None -> the file has no header line, treat row 0 as data.
    # usecols + names lets us read only the columns we need and rename them.
    reader = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,        # name ALL 125 columns
        usecols=wanted_idx,   # but only parse the ones we want
        chunksize=chunsize,
        low_memory=False,
        on_bad_lines="skip",  # a few malformed rows should never crash us
    )

    for chunk in reader:
        # pandas may reorder usecols; make the column order predictable.
        yield chunk[use_columns]


def select_targets(path, max_targets=200, tmag_max=12.0,
                   require_radius=True, chunsize=100_000, verbose=True):
    """
    Walk the catalogue and pick good candidate stars to look for transits in.

    The selection logic, and why each cut is here:

      * Tmag <= tmag_max : bright stars have higher signal-to-noise light
        curves, so a real transit is easier to see. Faint stars are mostly
        noise.
      * radius present and sane : we need the stellar radius to turn a transit
        depth into a planet radius later. No radius -> we cannot do the
        science, so skip.
      * finite ra/dec : we need coordinates to fetch the light curve.

    We stop once we have collected `max_targets` rows, so this returns quickly
    even though the file itself is enormous.

    Returns
    -------
    pandas.DataFrame
        The selected targets (at most max_targets rows).
    """
    collected = []
    n_seen = 0

    for chunk in iter_catalog(path, chunize=chunsize):
        n_seen += len(chunk)

        # numeric coercion -- catalogue has blank fields that read as NaN/str
        for col in ["ra", "dec", "Tmag", "rad", "Teff", "contratio"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        mask = chunk["ra"].notna() & chunk["dec"].notna()
        mask &= chunk["Tmag"].notna() & (chunk["Tmag"] <= tmag_max)
        if require_radius:
            mask &= chunk["rad"].notna() & (chunk["rad"] > 0)

        good = chunk[mask]
        if len(good):
            collected.append(good)

        have = sum(len(c) for c in collected)
        if verbose:
            print("  [catalog] scanned {:>9,} rows, kept {:>5} targets"
                  .format(n_seen, have), end="\r")

        if have >= max_targets:
            break

    if verbose:
        print()  # finish the \r line

    if not collected:
        return pd.DataFrame(columns=USE_COLUMNS)

    result = pd.concat(collected, ignore_index=True).head(max_targets)
    return result


if __name__ == "__main__":
    # Tiny self-test against a synthetic mini-catalogue so this file can be
    # checked without the real 10 GB download present.
    import io
    demo_rows = []
    for i in range(5):
        row = ["0"] * len(COLUMNS)
        row[0] = str(1000 + i)        # ID
        row[13] = str(150.0 + i)      # ra
        row[14] = str(-20.0 + i)      # dec
        row[60] = str(9.0 + i * 0.5)  # Tmag
        row[70] = str(1.0 + i * 0.1)  # rad
        demo_rows.append(",".join(row))
    demo_csv = "\n".join(demo_rows) + "\n"

    tmp = "/tmp/_mini_ctl.csv"
    with open(tmp, "w") as f:
        f.write(demo_csv)

    sel = select_targets(tmp, max_targets=3, tmag_max=12.0)
    print("self-test selected:")
    print(sel[["ID", "ra", "dec", "Tmag", "rad"]].to_string(index=False))
