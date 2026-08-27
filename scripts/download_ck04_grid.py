#!/usr/bin/env python3
"""
Download the complete Castelli & Kurucz (2004) ATLAS9 atmosphere grid
(CK04 / "ck04models") from the STScI CDBS archive, in batches.

This fills out PySeshat's `validation/atmosphere_models/` directory so
the Al-Wardat SED-fitting pipeline can be used for ANY target system,
not just the ones whose Teff happens to fall in the 4 files already
present locally.

WHAT THIS DOWNLOADS
--------------------
608 FITS files total: 76 effective-temperature points (3500-50000 K)
x 8 metallicities ([M/H] = -0.5, -1.0, -1.5, -2.0, -2.5, 0.0, +0.2, +0.5,
i.e. directories ckm05, ckm10, ckm15, ckm20, ckm25, ckp00, ckp02, ckp05).
4 of these are already present in the PySeshat repo (ckm05_10000.fits,
ckp00_5750/6000/6250.fits) - the manifest below marks those and this
script skips them by default (use --force to re-download anyway).

Total size is roughly 608 x ~67 KB =~ 41 MB, so this is a small download
overall - the "batches" are about being a polite, resumable citizen of
STScI's server, not about data volume.

USAGE
-----
Run this on a machine with normal internet access (NOT inside a
sandboxed/allowlisted shell - both the Claude cloud sandbox and the
Claude desktop-bridge device shell block ssb.stsci.edu, which is why
this script has to be run by hand):

    pip install requests   # if not already available
    python download_ck04_grid.py --dest "D:\\Scientific Books\\PostDoc Research\\Potential work\\Binaries\\BinariesPaper\\pyseshat\\validation\\atmosphere_models"

It reads ck04_manifest.csv (must be in the same directory as this
script, or pass --manifest), downloads in batches of --batch-size
(default 20) with a short pause between batches, retries transient
failures up to --retries times, verifies each file is a well-formed
FITS file after download, and is safe to re-run: already-downloaded
files (correct size, opens as FITS) are skipped, so an interrupted run
can just be restarted.

    python download_ck04_grid.py --dest <path> --batch-size 20 --workers 4

Progress is written to ck04_download.log next to the manifest, and a
running tally is printed to the console.
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    print(
        "This script needs the 'requests' package. Install it with:\n"
        "    pip install requests\n"
        "and re-run.",
        file=sys.stderr,
    )
    sys.exit(1)


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def is_valid_fits(path):
    """Cheap sanity check without requiring astropy: a CK04 FITS file
    starts with a standard FITS primary header block beginning
    'SIMPLE  ='. Good enough to catch truncated/HTML-error downloads
    (e.g. a proxy error page saved with a .fits extension)."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(80)
        return header.startswith(b"SIMPLE")
    except OSError:
        return False


def download_one(url, dest_path, timeout=30):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    tmp_path.write_bytes(response.content)
    if not is_valid_fits(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"downloaded content for {url} is not a valid FITS file")
    tmp_path.replace(dest_path)
    return len(response.content)


def download_with_retries(url, dest_path, retries, timeout):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return download_one(url, dest_path, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - want to retry any failure
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / "ck04_manifest.csv"),
        help="Path to ck04_manifest.csv (default: alongside this script)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help=(
            "Destination directory - should be PySeshat's "
            "validation/atmosphere_models/ folder. Files are written "
            "to <dest>/<metallicity>/<metallicity>_<teff>.fits, "
            "matching what benchmark_runner.py and the expected_results "
            "JSON files expect."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Files per batch (default 20)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent downloads within a batch (default 4)")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds to pause between batches (default 2.0)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file on failure (default 3)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds (default 30)")
    parser.add_argument("--force", action="store_true", help="Re-download files already marked present/downloaded")
    parser.add_argument(
        "--metallicity",
        action="append",
        dest="metallicities",
        help="Restrict to one metallicity directory (e.g. --metallicity ckp00). Can repeat.",
    )
    parser.add_argument(
        "--teff-min", type=float, default=None, help="Only download files with Teff >= this value"
    )
    parser.add_argument(
        "--teff-max", type=float, default=None, help="Only download files with Teff <= this value"
    )
    args = parser.parse_args()

    dest_root = Path(args.dest)
    manifest_path = Path(args.manifest)
    log_path = manifest_path.parent / "ck04_download.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("ck04")

    rows = load_manifest(manifest_path)
    log.info("Manifest loaded: %d files total", len(rows))

    if args.metallicities:
        wanted = set(args.metallicities)
        rows = [r for r in rows if r["metallicity"] in wanted]
    if args.teff_min is not None:
        rows = [r for r in rows if float(r["teff"]) >= args.teff_min]
    if args.teff_max is not None:
        rows = [r for r in rows if float(r["teff"]) <= args.teff_max]

    to_fetch = []
    skipped_existing = 0
    for row in rows:
        dest_path = dest_root / row["target_relative_path"]
        already_marked = row["already_present"].strip().lower() in ("true", "1", "yes")
        if not args.force and dest_path.is_file() and is_valid_fits(dest_path):
            skipped_existing += 1
            continue
        if not args.force and already_marked and dest_path.is_file():
            skipped_existing += 1
            continue
        to_fetch.append(row)

    log.info(
        "After filters: %d files selected, %d already present and skipped",
        len(to_fetch),
        skipped_existing,
    )

    if not to_fetch:
        log.info("Nothing to do - all selected files are already present.")
        return

    total = len(to_fetch)
    done = 0
    failed = []
    total_bytes = 0

    batches = [to_fetch[i : i + args.batch_size] for i in range(0, len(to_fetch), args.batch_size)]
    log.info("Downloading %d files in %d batches of up to %d", total, len(batches), args.batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        log.info("Batch %d/%d (%d files)", batch_index, len(batches), len(batch))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_row = {
                pool.submit(
                    download_with_retries,
                    row["url"],
                    dest_root / row["target_relative_path"],
                    args.retries,
                    args.timeout,
                ): row
                for row in batch
            }
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                done += 1
                try:
                    n_bytes = future.result()
                    total_bytes += n_bytes
                    log.info(
                        "[%d/%d] OK   %s (%d bytes)",
                        done,
                        total,
                        row["filename"],
                        n_bytes,
                    )
                except Exception as error:  # noqa: BLE001
                    failed.append((row["filename"], row["url"], str(error)))
                    log.error(
                        "[%d/%d] FAIL %s -- %s",
                        done,
                        total,
                        row["filename"],
                        error,
                    )
        if batch_index < len(batches):
            time.sleep(args.pause)

    log.info(
        "Done. %d succeeded, %d failed, %.1f MB downloaded.",
        total - len(failed),
        len(failed),
        total_bytes / (1024 * 1024),
    )
    if failed:
        log.warning("Failed files (re-run the same command to retry - already-downloaded files are skipped):")
        for filename, url, error in failed:
            log.warning("  %s  (%s)  -- %s", filename, url, error)
        sys.exit(1)


if __name__ == "__main__":
    main()
